# Set your HuggingFace HOME directory to store downloaded model and datasets, default is your own HOME directory.
export HF_HOME="your/HF_HOME_directory"

# 3 bits
python main.py --model path/to/llama-3-8b-hf --epochs 20 --output_dir ./log/llama-3-8b-w3a16g128-mod --eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc --lwc_lr 0.015 --aug_loss --datatype mod
# 4 bits
python main.py --model path/to/llama-3-8b-hf --epochs 20 --output_dir ./log/llama-3-8b-w4a16g128-mod --eval_ppl --wbits 4 --abits 16 --group_size 128 --lwc --lwc_lr 0.015 --aug_loss --datatype mod