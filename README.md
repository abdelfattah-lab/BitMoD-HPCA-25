# BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration [\[Paper\]](https://arxiv.org/abs/2411.11745)

BitMoD is an algorithm-hardware co-design framework for LLM acceleration using bit-serial hardware with mixture-of-datatypes. It supports diverse precision and data types with a flexible accuracy-efficiency trade-off. 

This repository contains the source code for reproducing the experiments of our HPCA'25 paper "BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration".

## News
- [2024/11] 🔥 BitMoD is accepted to [HPCA 2025](https://hpca-conf.org/2025/) !
- [2024/10] 🔥 We have extended [OmniQuant](https://github.com/OpenGVLab/OmniQuant/tree/main) to support BitMoD datatypes.
- [2024/07] 🔥 We have extended [AWQ](https://github.com/OpenGVLab/OmniQuant) and [SmoothQuant](https://github.com/mit-han-lab/smoothquant) to support BitMoD datatypes.

## Getting Started
Every folder in this repo is used for a separate set of experiments in the BitMoD paper. Please go to each folder and follow its `README` to run different experiments. 

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
