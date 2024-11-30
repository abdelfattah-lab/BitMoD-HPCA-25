## Installation

```bash
conda create -n smoothquant python=3.8
conda activate smoothquant
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.36.0 accelerate datasets zstandard

python setup.py install
```

## Running the experiments

```bash
./run_experiments.sh
```

The perplexity results will be saved in the file `results_mod/summary.csv` under this directory.

## Notes:

This folder is based on [SmoothQuant codebase](https://github.com/mit-han-lab/smoothquant.git)
