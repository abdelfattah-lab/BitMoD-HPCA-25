
# BitMoD hardware simulator 

This folder contains file and script to reproduce **_Fig. 7_** and **_Fig. 8_** of our BitMoD paper. 
To run the experiments, first change to this directory and activate the **awq-bitmod** conda environment. If you haven't set up the **awq-bitmod** environment, follow the instructions under the **AWQ-BitMoD** folder to set up the environment.
```
cd bitmod_sim
conda activate awq-bitmod
```

1. Please change the default HuggingFace directory in `run_shape_profile.sh`
```
export HF_HOME="your/HF_HOME/directory"
```

2. Profile the LLM configuration and layer shape. The profiled information will be saved in a new folder **model_shape_config** under this directory.
```
bash run_shape_profile.sh
```

3. Get the latency and energy of different models for discriminative and generative tasks.
```
python test_baseline.py --is_generation                  # Baseline FP16 accelerator
python test_ant.py      --is_generation                  # ANT accelerator
python test_olive.py    --is_generation                  # OliVe accelerator
python test_bitmod.py   --is_generation --is_lossless    # BitMoD accelerator
```
The flag `--is_generation` is optional. When enabled / disabled, it will evaluate the hardware performance of generative / discriminative tasks.
The flag `--is_lossless` is optional for BitMoD. When enabled / disabled, it will evaluate the hardware performance of lossless / lossy BitMoD quantization. Please see **_Section V-C_** of our paper for more details.

Note that the weight precision of ANT and OliVe are hard-coded in our simulator. 
The precision are profiled offline by ensring their quanperplexity and accuracy are acceptable after using their quantization data types and algorithms. 

4. To generate **_Fig. 7_** and **_Fig. 8_** of the BitMoD paper, go to `plot` directory and run the jupyter notebooks.
Note that the cycle and energy numbers are the same as those generated in Step 3.
