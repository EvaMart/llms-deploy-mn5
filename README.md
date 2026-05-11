# Running LLMs in Mare Nostrum 5

Tests, checks and notes about working with models in MN5.

## Documentation 
https://www.bsc.es/supportkc/docs/MareNostrum5/intro/ 


## Data Transfer

* Use `/gpfs/projects` or `/gpfs/scratch` for big files (like models).  
* Use transfer nodes to copy big files from local. For example: 
    ```sh
    scp -r models/<model-name> bscXXXX@transfer1.bsc.es:/gpfs/home/bsc/bscXXXX/models/<model-name>
    ```

## Environment

- Use `login4` to install packages from the internet. 
- Always load modules intel mkl impi hdf5 to be able to load Python 

### Installing/upgrading Python modules 

Example to upgrade some modules 
```sh
module purge
module load intel mkl impi hdf5 python/3.12.1
source /gpfs/home/bsc/bsc619075/envs/llm/bin/activate
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install "transformers>=5.8" accelerate safetensors sentencepiece
python -m pip install --upgrade "transformers>=4.51.0" accelerate safetensors tokenizers
```

### Ensuring `pip` used is the one in the environment: 

```sh
source /gpfs/home/bsc/bsc619075/envs/llm/bin/activate

export PYTHONNOUSERSITE=1
unset PYTHONPATH

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
``` 
