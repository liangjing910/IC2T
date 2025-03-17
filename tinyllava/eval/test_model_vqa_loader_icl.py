import os
import json
from model_vqa_loader_icl import eval_model  # 请确保model_vqa_loader_icl.py在当前路径或PYTHONPATH中
import torch

# 创建一个模拟的参数对象
class Args:
    model_path = "/home/LiangJing/TinyLLaVA_Factory/checkpoints/llava_factory/custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora/"
    model_base = "custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora"
    image_folder = "/home/LiangJing/TinyLLaVA_Factory/dataset/eval/gqa/data/images"
    question_file = "/home/LiangJing/TinyLLaVA_Factory/dataset/eval/gqa/llava_gqa_testdev_balanced_icl.jsonl"
    answers_file = "/home/LiangJing/TinyLLaVA_Factory/dataset/eval/gqa/answers/test_answers.jsonl"
    conv_mode = "phi"
    num_chunks = 1
    chunk_idx = 0
    temperature = 0.0
    top_p = None
    num_beams = 1
    max_new_tokens = 128

# 创建参数实例
args = Args()

# 调用 eval_model 函数
eval_model(args)