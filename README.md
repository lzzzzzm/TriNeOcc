## TriNeOcc

### Installation

**Step 0. Pyotrch.**
```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

**Step 1. Install MMEngine, MMCV and MMDetection using MIM.**
```bash
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0rc4
mim install mmdet==3.0.0
```

**Step 2. Install Project.**
```bash
pip install -v -e .
```

**Step 3. Install tiny-cuda-nn and nerfacc.**

install ninja
```bash
pip install ninja
```
**Windows**
```bash
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
# modify the tiny-cud-nn/bingdings/torch/setup.py the last line to
# cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
# use Command Prompt for VS 2019 to run
activate environment_name
cd VC\Auxiliary\Build\
vcvarsall.bat x64
```

```bash
cd {project_path}/tiny-cuda-nn/bindings/torch
python setup.py install
```
```bash
pip install nerfacc==0.5.3 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.0_cu116.html
```