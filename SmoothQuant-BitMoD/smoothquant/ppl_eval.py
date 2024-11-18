import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model
import tqdm

from datasets import load_dataset
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument(
    "--act_scales_path",
    type=str,
    default="act_scales/llama-2-7b.pt",
)
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--results_path", type=str)
parser.add_argument("--group_size", type=int, default=128)
parser.add_argument("--wquantization", type=str, default="per_channel")
parser.add_argument("--datatype", type=str, default="wrong")
parser.add_argument("--wq_bits", type=int, default=4)


args = parser.parse_args()
alpha = args.alpha
model_path = args.model_path
act_scales_path = args.act_scales_path
n_samples = args.n_samples


class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))


tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=n_samples)

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)

if args.smooth:
    act_scales = torch.load(act_scales_path)
    smooth_lm(model, act_scales, alpha)
if args.quantize:
    model = quantize_model(
        model,
        # weight_quant="per_channel",
        weight_quant=args.wquantization,
        act_quant="per_token",
        group_size=args.group_size,
        quantize_bmm_input=True,
        datatype=args.datatype,
        wq_bits=args.wq_bits,
    )

ppl = evaluator.evaluate(model)
print(f"Perplexity: {ppl}")

# Save results in results_path
with open(args.results_path, "w") as f:
    f.write(f"Perplexity: {ppl}")
    f.write("\n")
