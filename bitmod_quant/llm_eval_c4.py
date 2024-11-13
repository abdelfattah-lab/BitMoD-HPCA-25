import torch
import torch.nn as nn
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

# Added by Yuzong Chen (yc2367@cornell.edu)
torch.set_grad_enabled(False)

model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='auto')
quant_model(model, wq_bits, wq_datatype, wq_groupsize)

random.seed(0)
model_net = model_str.split('/')[-1]
model_family = '_'.join(model_net.lower().split('-')[:-1])
model.seqlen = 2048

cache_testloader = f'./data_cache//testloader_{model_family}_c4_{model.seqlen}.cache'
os.makedirs(os.path.dirname(cache_testloader), exist_ok=True)
if os.path.exists(cache_testloader):
    testenc = torch.load(cache_testloader)
    print(f"load calibration from {cache_testloader}")
else:
    valenc = []
    enc = AutoTokenizer.from_pretrained(model_str, use_fast=False, trust_remote_code=True)
    testenc = load_dataset("allenai/c4", data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split="validation")
    for _ in range(256): # run 256 samples
        while True:
            i = random.randint(0, len(testenc) - 1)
            tmp = enc(testenc[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] > (model.seqlen + 1):
                break
        i = random.randint(0, tmp.input_ids.shape[1] - model.seqlen - 1)
        j = i + model.seqlen
        valenc.append(tmp.input_ids[:, i:j])
    testenc = torch.hstack(valenc)
    torch.save(testenc, cache_testloader)

nsamples = testenc.numel() // model.seqlen
loss_fct = nn.CrossEntropyLoss()
nlls = []
with tqdm(range(nsamples)) as progress:
    for i in progress:
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:].to(model.device)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood.item())
        progress.set_description(f"Evaluating")

ppl = torch.exp(torch.tensor(nlls).sum() / (nsamples * model.seqlen))
print(f'c4 perplexity: {ppl}')
print('\n')

write_results(ppl.item(), model_str, 'c4', wq_bits, wq_datatype, wq_groupsize)
