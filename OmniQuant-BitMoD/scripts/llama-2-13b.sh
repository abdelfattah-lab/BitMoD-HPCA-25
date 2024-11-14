# 3 bits
python main.py --model path/to/llama-2-13b-hf --epochs 20 --output_dir ./log/llama-2-13b-hf-w3a16g128 --eval_ppl --wbits 3 --abits 16 --group_size 128 --lwc --lwc_lr 0.015 --aug_loss --datatype mod
# 4 bits
python main.py --model path/to/llama-2-13b-hf --epochs 20 --output_dir ./log/llama-2-13b-hf-w4a16g128 --eval_ppl --wbits 4 --abits 16 --group_size 128 --lwc --lwc_lr 0.015 --aug_loss --datatype mod