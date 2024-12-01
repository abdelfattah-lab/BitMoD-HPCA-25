
# Combining BitMoD data types with AWQ optimization

This folder is built upon the official [AWQ](https://github.com/mit-han-lab/llm-awq) repository, and contains file and script to reproduce the results in **_Table XI_** of our BitMoD paper. 
To run the experiments, first change to this directory and install the **awq-bitmod** conda environment. 
```bash
cd AWQ-BitMoD
conda create -n awq-bitmod python=3.10 -y
conda activate awq-bitmod
pip install --upgrade pip  # enable PEP 660 support
python -m pip install -e .
```
Also, follow the official AWQ repository, install the efficient W4A16 (4-bit weight, 16-bit activation) CUDA kernel and optimized FP16 kernels (e.g. layernorm, positional encodings).
```bash
cd awq/kernels
python setup.py install
```

## Running AWQ with different data types
1. Perform AWQ search and save search results 
```bash
python -m awq.entry --model_path /PATH/TO/HF_MODEL \
    --w_bit 3 \
    --q_group_size 128 \
    --wq_dtype "bitmod" \
    --run_awq --dump_awq "./awq_cache/${model_name}-w3-bitmod.pt"
```
Here, `w_bit` is the quantization precision (3 or 4), `q_group_size` is the quantization group size (we use 128 by default), `wq_dtype` is the quantization data type (`int` or `bitmod`).
Note that the `${model_name` variable is used to specify the saved AWQ cache file. We suggest using distinguishable names (e.g. "llama-2-7b", "llama-3-8b") for every model.

2. Evaluate Wikitext and C4 perplexity using the pre-computed AWQ results
```bash
python -m awq.entry --model_path ${model_path} \
    --w_bit 3 \
    --q_group_size 128 \
    --wq_dtype "bitmod" \
    --load_awq "./awq_cache/${model_name}-w3-bitmod.pt" \
    --q_backend fake \
    --eval_ppl \
    --output_path "./results/${model_name}/ppl/w3-bitmod"
```
The perplexity results will be saved in the folder `results` under this directory.

3. We also provide automatic scripts `run_awq.sh` and `run_eval_ppl.sh` to run experiments on three models `Llama-2-7B, Llama-2-13B, Llama-3-8B`, two precision `3, 4`, and two data types `int, bitmod` to reproduce all AWQ perplexity in **_Table XI_** of our BitMoD paper.
   
    3.1\) Please follow the instructions in `run_awq.sh` and `run_eval_ppl.sh` to change the default HuggingFace directory  
    ```bash
    export HF_HOME="your/HF_HOME/directory"
    ```
    
    3.2\) Then run the shell scripts
    ```bash
    bash run_awq.sh  # will take several hours
    bash run_eval_ppl.sh
    ```
The perplexity result will be saved in a folder called `results` under this directory.
