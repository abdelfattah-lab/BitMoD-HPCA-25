# BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration [\[Paper\]](https://arxiv.org/abs/2411.11745)

BitMoD is an algorithm-hardware co-design framework for LLM acceleration using bit-serial hardware with mixture-of-datatypes. It supports diverse precision and data types with a flexible accuracy-efficiency trade-off. 

This repository contains the source code for reproducing the experiments of our HPCA'25 paper "BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration".

## News
- [2024/11] 🔥 BitMoD is accepted to [HPCA 2025](https://hpca-conf.org/2025/) !
- [2024/10] 🔥 We have extended [OmniQuant](https://github.com/OpenGVLab/OmniQuant/tree/main) to support BitMoD datatypes.
- [2024/07] 🔥 We have extended [AWQ](https://github.com/OpenGVLab/OmniQuant) and [SmoothQuant](https://github.com/mit-han-lab/smoothquant) to support BitMoD datatypes.

## Getting Started
Every folder in this repo is used for a separate set of experiments in the BitMoD paper. Please go to each folder and follow its `README` to run different experiments. 

## Perplexity Results
| Model  | Precision | Quant Method | WikiText2 PPL | C4 PPL |
| ------ | --------- | ------------ | ------------- | ------ |
|  Llama-2-7B  |fp16   |                | 5.47  | 6.97 |
|  Llama-2-7B  |w4g128 | AWQ   + BitMoD | 5.59 | 7.09 |
|  Llama-2-7B  |w4g128 | OmniQ + BitMoD | 5.56 | 7.06 |
|  Llama-2-7B  |w3g128 | AWQ   + BitMoD | 6.07 | 7.64 |
|  Llama-2-7B  |w3g128 | OmniQ + BitMoD | 5.86 | 7.55 |
|  Llama-2-13B |fp16   |                | 4.88 | 6.47 |
|  Llama-2-13B |w4g128 | AWQ   + BitMoD | 4.96 | 6.55 |
|  Llama-2-13B |w4g128 | OmniQ + BitMoD | 4.95 | 6.55 |
|  Llama-2-13B |w3g128 | AWQ   + BitMoD | 5.27 | 6.88 |
|  Llama-2-13B |w3g128 | OmniQ + BitMoD | 5.17 | 6.84 |
|  Llama-2-70B |fp16   |                | 3.32 | 5.52 |
|  Llama-2-70B |w4g128 | AWQ   + BitMoD | 3.40 | 5.57 |
|  Llama-2-70B |w3g128 | AWQ   + BitMoD | 3.70 | 5.77 |
|  Llama-3-8B  |fp16   |                | 6.14 | 8.88 |
|  Llama-3-8B  |w4g128 | AWQ   + BitMoD | 6.50 | 9.32 |
|  Llama-3-8B  |w4g128 | OmniQ + BitMoD | 6.45 | 9.34 |
|  Llama-3-8B  |w3g128 | AWQ   + BitMoD | 7.79 | 11.07 |
|  Llama-3-8B  |w3g128 | OmniQ + BitMoD | 7.56 | 11.06 |
|  Llama-3-70B |fp16   |                | 2.85 | 6.73 |
|  Llama-3-70B |w4g128 | AWQ   + BitMoD | 3.19 | 6.94 |
|  Llama-3-70B |w3g128 | AWQ   + BitMoD | 4.48 | 7.76 |

## Code Structure
```
Repo Root
|---- SmoothQuant-BitMoD   # Running SmoothQuant with basic INT and our proposed BitMoD data types
|---- AWQ-BitMoD           # Running AWQ with basic INT and our proposed BitMoD data types
|---- OmniQuant-BitMoD     # Running OmniQuant with basic INT and our proposed BitMoD data types
|---- bitmod-quant         # Weight-only quantization with different precision and data types (e.g. INT, FP, BitMoD)
|---- bitmod-sim           # BitMoD accelerator simulator
```

## Citation
```bibtex
@article{chen2025hpca,
  title={{BitMoD}: Bit-serial Mixture-of-Datatype LLM Acceleration},
  author={Yuzong Chen and Ahmed F. AbouElhamayed and Xilai Dai and Yang Wang and Marta Andronic and George A. Constantinides and Mohamed S. Abdelfattah},
  journal={IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
  year={2025}
}
```

-----------------

_This work is subject to a patent application filed by Cornell University._
