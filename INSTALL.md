### Set up the python environment

```
conda create -n ssnake python=3.8
conda activate ssnake

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 11.3, install torch 1.10.0 built from cuda 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch

# install dependencies
pip install -r requirements.txt
```

### Compile cuda extensions under `lib/csrc`

```
ROOT=/path/to/ssnake
cd $ROOT/lib/csrc
export CUDA_HOME="/usr/local/cuda-11.3"
cd ./extreme_utils
python setup.py build_ext --inplace

```
