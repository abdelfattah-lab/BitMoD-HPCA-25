import os 

def write_results(ppl: float, model_name: str, dataset: str, wq_bits: int, wq_datatype: str, wq_groupsize: int):
    BASE_PATH = "./results_quant"
    if wq_groupsize <= 0:
        wq_groupsize = "none"
    
    model_info = model_name.split('/')
    if len(model_info) > 1:
        model_dir = f"{model_info[0]}__{model_info[1]}"
    else:
        model_dir = f"{model_info[0]}__"
    dataset_dir = f"{dataset}"
    wq_dir      = f"w_{wq_bits}_gs_{wq_groupsize}"
    dtype_file  = f"{wq_datatype}.txt"
    if (wq_datatype == 'fp16'):
        output_path = f'{BASE_PATH}/{dataset_dir}/{model_dir}/{dtype_file}'
    else:
        output_path = f'{BASE_PATH}/{dataset_dir}/{model_dir}/{wq_dir}/{dtype_file}'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.writelines(f'{dataset} perplexity: {ppl} \n')
    
    print('Successfully written results. \n\n')