import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import random, time, argparse, os
from tqdm import tqdm
from typing import Optional

from quant_utils.quant_weight import quant_model
from quant_utils.write_results import write_results


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
)
parser.add_argument(
    "--wq_datatype", type=str, default="", help="The weight datatype for weight-only quantization",
)
parser.add_argument(
    "--wq_bits", type=int, default=4, help="The weight precision for weight-only quantization",
)
parser.add_argument(
    "--wq_groupsize", type=int, default=None, help="The quantization group size for weight-only quantization",
)
args = parser.parse_args()
model_str = args.model
wq_bits = args.wq_bits
wq_groupsize = args.wq_groupsize
wq_datatype = args.wq_datatype

torch.set_grad_enabled(False)
torch.manual_seed(0)
random.seed(0)

model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
quant_model(model, wq_bits, wq_datatype, wq_groupsize)
model.seqlen = 2048
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_str, use_fast=False, trust_remote_code=True)
testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
testenc = testenc.input_ids.to(model.device)
nsamples = testenc.numel() // model.seqlen
loss_fct = torch.nn.CrossEntropyLoss()

nlls = []
for i in tqdm(range(nsamples), desc="evaluating..."):
    batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
        model.device
    )
    with torch.no_grad():
        lm_logits = model(batch).logits
    shift_logits = lm_logits[:, :-1, :].contiguous().float()
    shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    neg_log_likelihood = loss.float() * model.seqlen
    nlls.append(neg_log_likelihood)

ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
print(f'wikitext perplexity: {ppl.item()}')

write_results(ppl.item(), model_str, "wikitext", wq_bits, wq_datatype, wq_groupsize)
