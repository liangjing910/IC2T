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
