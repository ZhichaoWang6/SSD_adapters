## SFT phase
We use [ms-swift](https://github.com/modelscope/ms-swift) with some modifications in the SFT phase to train [MMDuet2](../README.md). 
When loading model (Qwen2.5-VL) and data, we use [qwen-vl-utils](https://pypi.org/project/qwen-vl-utils/) and [transformers](https://github.com/huggingface/transformers). 


## Training Process
### Create conda environment
```bash
conda create --name mmduet2_sft python=3.10
conda activate mmduet2_sft
pip install -r requirements.txt
```

### Download and replace some code files ms-swift
```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout v3.2.0
pip install -e .
cd ..

# replace some code files that we modified from ms-swift
cp -ri ms-swift-replace-code/* ms-swift
```

### Download training data
Follow the instructions in [MMDuet2-data](https://huggingface.co/datasets/wangyueqian/MMDuet2-data) to prepare the dataset, and move the `sft` sub folder to `./data/annotations`

### Train! 
```bash
bash ./scripts/train.sh
```
