# IC2T: Embedding In-Context Learning into Chain of Thought for Vision-Language Models

## Overview
This repository contains the official implementation of the paper "**IC2T: Embedding In-Context Learning into Chain of Thought for Vision-Language Models**".

**IC2T** is a novel multimodal reasoning framework designed to progressively optimize multimodal reasoning by integrating local visual details and global contextual features through a multi-round reasoning process. The framework significantly enhances model performance in complex multimodal tasks like Visual Question Answering (VQA) and text-image alignment.




## Setup

### Prerequisites
- Python >= 3.10
- PyTorch
- CUDA-enabled GPU (recommended)

### Installation
```bash
conda create -n ic2t python=3.10 -y
conda activate ic2t
pip install --upgrade pip  # enable PEP 660 support
pip install -e 
```

## Dataset
Due to size constraints, only a sample of the IC2T dataset (ic2t_dataset_sample.json) is provided in this repository. The complete dataset used in the paper is too large for direct inclusion. Please refer to the sample file for dataset structure and format. For the complete dataset, please contact the authors directly. The complete dataset will be made available via an external link shortly.


🚀 **Continuous Updates**
The **IC2T** codebase, datasets, and pretrained model weights will be continuously updated. Stay tuned for further improvements and releases!

## Acknowledgments
This implementation is based on the [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory). We gratefully acknowledge their contribution.
