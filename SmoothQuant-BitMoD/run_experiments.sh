#!/bin/bash

# Set your HuggingFace HOME directory to store downloaded model and datasets, default is your own HOME directory.
export HF_HOME="your/HF_HOME/directory"

mkdir -p results_mod

models=(
  "meta-llama/Meta-Llama-3-8B"
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
)
wq_bits_vs=(4 3)
group_size_vs=(128 64)
datatype_vs=(mixed_bitmod)
alpha=0.85

# Create CSV header
echo "model,group_size,wq_bits,datatype,perplexity" > results_mod/summary.csv

for model in "${models[@]}"; do
    for wq_bits in "${wq_bits_vs[@]}"; do
        for group_size in "${group_size_vs[@]}"; do
            for datatype in "${datatype_vs[@]}"; do
                model_name=${model##*/}
                echo $model_name
                wquantization=mod
                act_scales=act_scales/$model_name.pt
                [ ! -e $act_scales ] && python examples/generate_act_scales.py --model-name $model --output-path $act_scales
                python smoothquant/ppl_eval.py \
                    --alpha $alpha \
                    --act_scales_path $act_scales \
                    --model_path $model \
                    --quantize --smooth --wquantization $wquantization --group_size $group_size \
                    --wq_bits $wq_bits --datatype $datatype \
                    --results_path results_mod/$model_name-$wquantization-$group_size-$wq_bits-$datatype.txt
                
                # Extract perplexity from results file and append to CSV
                result_file="results_mod/$model_name-$wquantization-$group_size-$wq_bits-$datatype.txt"
                if [ -f "$result_file" ]; then
                    perplexity=$(grep -o 'Perplexity: [0-9.]*' "$result_file" | cut -d' ' -f2)
                    echo "$model_name,$group_size,$wq_bits,$datatype,$perplexity" >> results_mod/summary.csv
                fi
            done
        done
    done
done
