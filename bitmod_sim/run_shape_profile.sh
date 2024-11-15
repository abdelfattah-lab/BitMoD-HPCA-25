#!/bin/bash

# Set your HuggingFace HOME directory to store downloaded model and datasets, default is your own HOME directory.
export HF_HOME="your/HF_HOME/directory"

declare -a model_list=("facebook/opt-1.3b" "microsoft/phi-2" "01-ai/Yi-6B" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Meta-Llama-3-8B")
for model in "${model_list[@]}"
do
    echo "model = ${model}"
    python llm_shape_profile.py --model ${model} 
done

