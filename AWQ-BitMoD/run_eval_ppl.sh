#!/bin/bash

# Set your HuggingFace HOME directory to store downloaded model and datasets, default is your own HOME directory.
export HF_HOME="/share/abdelfattah/temp_yc2367/huggingface"

# This is the model name list for running AWQ
# NOTE: Every model name should only contain alphabet numbers and the dash symbol, since the model name will be used to specify the awq-cache file
# Hence, for every model name you should also set the "model_path" variable in the following for-loop, which is the path to load the HF model
declare -a model_name_list=("llama-3-8b" "llama-2-7b" "llama-2-13b")
# Set the quantization group size. Use -1 for per-channel quantization
group_size=128
# Set the quantization precision list.
w_bit_list=(4 3)
# Set the datatype list. We only evaluate INT and BitMoD
declare -a datatype_list=("int" "bitmod")

for w_bit in "${w_bit_list[@]}"
do
    for model_name in "${model_name_list[@]}"
    do
        # Set the model path for every model name you specified in "model_name_list" above
        if [[ ${model_name} == "llama-2-7b" ]]
        then
            model_path="meta-llama/Llama-2-7b-hf"
        elif [[ ${model_name} == "llama-2-13b" ]]
        then
            model_path="meta-llama/Llama-2-13b-hf"
        elif [[ ${model_name} == "llama-3-8b" ]]
        then
            model_path="meta-llama/Meta-Llama-3-8B"
        fi
        
        for wq_datatype in "${datatype_list[@]}"
        do
            awq_cache_path="./awq_cache/${model_name}-w${w_bit}-g${group_size}-${wq_datatype}.pt"
            # Change to result_save_path="./results/${model_name}/acc/w${w_bit}-${wq_datatype}" if you want to evaluate downstream task accuracy
            result_save_path="./results/${model_name}/ppl/w${w_bit}-${wq_datatype}"

            echo "#################### Running Experiment ####################"
            echo "Model name        = ${model_name}"
            echo "Model path        = ${model_path}"
            echo "Quant precision   = ${w_bit}"
            echo "Quant group size  = ${group_size}"
            echo "Quant datatype    = ${wq_datatype}"
            echo "AWQ cache path    = ${awq_cache_path}"
            echo "Result save path  = ${result_save_path}"
            echo "############################################################"
            echo 

            echo "Running evaluation"
            if [ ${w_bit} = 16 ] 
            then
                python -m awq.entry --model_path ${model_path} \
                    --eval_ppl --output_path ${result_save_path} 
            else
                python -m awq.entry --model_path ${model_path} \
                    --w_bit ${w_bit} --q_group_size ${group_size} --wq_dtype ${wq_datatype} \
                    --load_awq ${awq_cache_path} --q_backend fake \
                    --eval_ppl --output_path ${result_save_path} 
            fi
        done
    done
done

