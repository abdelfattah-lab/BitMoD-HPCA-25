#!/bin/bash

# Set your HuggingFace HOME directory to store downloaded model and datasets, default is your own HOME directory.
export HF_HOME="your/HF_HOME/directory"

# Modify the model path to your own path. 
declare -a model_list=("facebook/opt-1.3b" "microsoft/phi-2" "01-ai/Yi-6B" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Meta-Llama-3-8B" )
# Set dataset: "wikitext" or "c4"
dataset="c4"
# Set the quantization precision list.
wq_bit_list=(16 3 4 5 6)
# Set the quantization group size. Use -1 for per-channel quantization
wq_groupsize=128

for wq_bit in "${wq_bit_list[@]}"
do
    # Change the datatype list to run quantization with different datatypes
    if [ ${wq_bit} = 3 ] 
    then
        declare -a datatype_list=("int3" "int3_asym" "mx_fp3" "mixed_ant" "mixed_er" "mixed_ea" "mixed_bitmod")
    elif [ ${wq_bit} = 4 ] 
    then 
        declare -a datatype_list=("int4" "int4_asym" "mx_fp4" "mixed_ant" "mixed_er" "mixed_ea" "mixed_bitmod")
    elif [ ${wq_bit} = 5 ] 
    then 
        declare -a datatype_list=("int5" "int5_asym" "fp5_e2m2" "fp5_e3m1")
    elif [ ${wq_bit} = 6 ] 
    then 
        declare -a datatype_list=("int6" "int6_asym" "fp6_e2m3" "fp6_e3m2")
    elif [ ${wq_bit} = 16 ] 
    then 
        declare -a datatype_list=("fp16")
    fi

    ## now loop through the above array
    for model in "${model_list[@]}"
    do
        for datatype in "${datatype_list[@]}"
        do  
            echo "#################### Running Experiment ####################"
            echo "Model             = ${model}"
            echo "Dataset           = ${dataset}"
            echo "Quant precision   = ${wq_bit}"
            echo "Quant group size  = ${wq_groupsize}"
            echo "Quant datatype    = ${datatype}"
            echo "############################################################"
    
            if [ ${dataset} = "wikitext" ] 
            then
                python llm_eval_wikitext.py \
                    --model ${model} \
                    --wq_bits ${wq_bit} \
                    --wq_datatype ${datatype} \
                    --wq_groupsize ${wq_groupsize}
            elif [ ${dataset} = "c4" ] 
            then     
                python llm_eval_c4.py \
                    --model ${model} \
                    --wq_bits ${wq_bit} \
                    --wq_datatype ${datatype} \
                    --wq_groupsize ${wq_groupsize}
            fi
        done
    done
done
