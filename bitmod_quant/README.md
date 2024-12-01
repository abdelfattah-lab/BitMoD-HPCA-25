
# Weight-only quantization using different datatypes

This folder contains the file and script to reproduce the results in **_Table VI_** and **_Table VIII_** of our BitMoD paper. 
To run the experiments, first change to this directory and activate the **awq-bitmod** conda environment. If you haven't set up the **awq-bitmod** environment, follow the instructions under the `AWQ-BitMoD` folder to set up the environment.
```
cd bitmod_quant
conda activate awq-bitmod
```
Evaluation of Wikitext perplexity
```
python llm_eval_wikitext.py \
  --model ${model} \
  --wq_bits ${quant_precision} \
  --wq_datatype ${quant_datatype} \
  --wq_groupsize ${quant_groupsize}
```
Evaluation of C4 perplexity
```
python llm_eval_c4.py \
  --model ${model} \
  --wq_bits ${quant_precision} \
  --wq_datatype ${quant_datatype} \
  --wq_groupsize ${quant_groupsize}
```
where `--model` is the model path to load the LLM, `--wq_bits` is an integer to specify the quantization precision, `wq_datatype` is the quantization data type (see below for supported data types), `--wq_groupsize` is the quantization group size (we use 128 by default).

We also provide an automatic script **run_exp.sh** to run experiments on different models, data types and precision. 
Please first change the default HuggingFace directory at the beginning of `run_exp.sh`
```
export HF_HOME="your/HF_HOME/directory"
```
Then you can modify `run_exp.sh` (e.g., model_list, datatype_list) and run massive experiments with
```
bash run_exp.sh
```
The perplexity results will be saved in a folder called `results_quant` under this directory.

## Supported Data Types:
The following table shows the supported data types
|  **Data Type**              | Definition                                     |
| --------------------------- | ---------------------------------------------- |
|  **fp16**                   | The baseline FP16 model without quantization   |
|  **int{n}**                 | Symmetric integer quantization, {n} should be replaced by the precision (e.g., **int4**) |
|  **int{n}_asym**            | Asymmetric integer quantization, {n} should be replaced by the precision (e.g., **int4_asym**) |
|  **fp5_e2m2**, **fp5_e3m1** | FP5 datatype with 2-bit exponent 2-bit mantissa, or 3-bit exponent 1-bit mantissa |
|  **fp6_e2m3**, **fp6_e3m2** | FP6 datatype with 2-bit exponent 3-bit mantissa, or 3-bit exponent 2-bit mantissa |
|  **mx_fp3**, **mx_fp4**     | [Microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) with element datatype **fp3** and **fp4**, respectively |
|  **mixed_ant**              | [ANT](https://arxiv.org/abs/2208.14286) datatype, currently support 3-bit and 4-bit ANT |
|  **mixed_er**               | FP4 and FP3 datatype with extra resolution, i.e., FP4_ER and FP3_ER in our BitMoD paper |
|  **mixed_ea**               | FP4 and FP3 datatype with extra asymmetry, i.e., FP4_EA and FP3_EA in our BitMoD paper |
|  **mixed_bitmod**           | Our 4-bit and 3-bit BitMoD datatype with both extra resolution and extra asymmetry |

## Supported Precision:
The following table shows the supported data types at different precision
|  **Precision**              | Supported Data Type                            |
| --------------------------- | ---------------------------------------------- |
|  16                   | fp16   |
|  8                    | int8, int8_asym   |
|  7                    | int7, int7_asym   |
|  6                    | int6, int6_asym, fp6_e2m3, fp6_e3m2   |
|  5                    | int5, int5_asym, fp5_e2m2, fp5_e3m1   |
|  4                    | int4, int4_asym, mx_fp4, mixed_ant, mixed_er, mixed_ea, mixed_bitmod   |
|  3                    | int3, int3_asym, mx_fp3, mixed_ant, mixed_er, mixed_ea, mixed_bitmod   |
