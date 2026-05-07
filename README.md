# CMSC 472 Final Project

## Overview
The goal of the project is to create a synthetic segmentation dataset. The success is measured by mean intersection over union (mIoU) of a Deeplab v3 model trained on the dataset. See the report for more information.

## Setup

### Python Versions and Dependencies
We use Python 3.9.25. There are two primary ways to install the necessary dependences.

1. `uv sync`: This is preferred, as it should handle any possible conflicts automatically. 
2. `pip install -r requirements.txt`: This is included just in case, but it is possible that packages don't resolve with this.

Everything in this repository requires NVIDIA GPUs to run (and is set up to handle multiple GPUs when appropriate).


### Data Download
Dataset download should happen automatically when running `gen_data_voc_llava_ngpu.sh` (see below). It downloads a HuggingFace dataset which consists of a small subset of the original PASCAL VOC challenge.

Our project involves the generation of a synthetic dataset. A saved generated dataset is available to download [here](https://drive.google.com/file/d/1je4BbkiT90iXEfyCNdb759e8PIxqyZOv/view?usp=sharing).

## Usage Instructions

Scripts to run the method are provided in `./scripts`. The hypothetical workflow goes as follows:
```bash
source scripts/gen_llava_captions_ngpu.sh
source scripts/gen_data_voc_llava_ngpu.sh
source scripts/apply_dense_crf_voc.sh
```

To evaluate results, we use a modified version of the evaluation script provided by the SyntaGen competition. The modified version of the script is accessible [here](https://colab.research.google.com/drive/1eXitNy_Z_PwO_1S0Jvxn07IWf3qfk0nc).

**Important**: Due to dependency issues in the original evaluation script, the new evaluation script requires very specific dependencies to be used in colab. The runtime version must be set to `2025.07` (`Runtime` &rarr; `Change Runtime Type`). Additionally, it will take two runs for packages to be installed properly. See the notebook for more details.


Some quicker debug utilities are included to make sure everything is working before committing to any long runs. These run using:
```bash
source scripts/gen_llava_captions_debug.sh
source scripts/gen_data_voc_llava_debug_2gpu.sh
```