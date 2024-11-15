# BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration

BitMoD is an algorithm-hardware co-design framework for LLM acceleration using bit-serial hardware with mixture-of-datatypes. It supports diverse precision and data types with a flexible accuracy-efficiency trade-off. 

This repository contains the source code for reproducing the experiments of our HPCA'25 paper "BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration".

## News
- [2024/11] 🔥 BitMoD is accepted to [HPCA 2025](https://hpca-conf.org/2025/) !
- [2024/10] 🔥 We have extended BitMoD to support [OmniQuant](https://github.com/OpenGVLab/OmniQuant/tree/main) with custom data types.
- [2024/07] 🔥 We have extended BitMoD to support [AWQ](https://github.com/OpenGVLab/OmniQuant) with custom data types.

## Getting Started
Every folder in this repo is used for a separate set of experiments in our paper. Please go to each folder and follow its `README` to run different experiments. 

## Code Structure
```
Repo Root
|---- AWQ-BitMoD           # Running AWQ with basic INT and our proposed BitMoD data types
|---- OmniQuant-BitMoD     # Running OmniQuant with basic INT and our proposed BitMoD data types
|---- bitmod-quant         # Weight-only quantization with different precision and data types (e.g. INT, ANT, BitMoD)
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
