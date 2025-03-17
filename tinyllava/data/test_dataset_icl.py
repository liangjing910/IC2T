import json
import torch
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tinyllava.data.dataset_icl import LazySupervisedDataset, DataCollatorForSupervisedDataset
from PIL import Image
import os
from tqdm import tqdm  # 导入 tqdm 库以显示进度条

# 定义数据集和图像路径
DATASET_PATH = "/home/LiangJing/TinyLLaVA_Factory/dataset/converted_gqa_output_v7.json"
IMAGE_PATH = "/home/LiangJing/TinyLLaVA_Factory/dataset"

# 从文件中加载数据集
with open(DATASET_PATH, "r") as f:
    dataset_data = json.load(f)

# 模拟 DataArguments 配置类
@dataclass
class DataArguments:
    data_path: str = DATASET_PATH
    image_folder: str = IMAGE_PATH  # 图像文件夹路径
    is_multimodal: bool = True
    image_processor: object = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")  # 图像处理器
    conv_version: str = "phi"  # 对话版本号
    image_aspect_ratio: str = "pad"  # 图像纵横比设置

# 初始化Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 使用适当的Tokenizer

# 创建数据集和整理器
data_args = DataArguments()
dataset = LazySupervisedDataset(data_path=DATASET_PATH, tokenizer=tokenizer, data_args=data_args)
collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

# 使用 DataLoader 测试数据集
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)

# 定义一个函数来检查图像文件的有效性
def is_image_file_valid(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # 检查图像文件是否损坏
        return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image file at {image_path}: {e}")
        return False

# 测试加载并检查图像文件
for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):  # 使用 tqdm 显示进度条
    # print(f"\nProcessing batch {i+1}...")
    # print("Batch input_ids:", batch["input_ids"])
    # print("Batch attention_mask:", batch["attention_mask"])
    # print("Batch labels:", batch["labels"])

    # 检查图像文件
    if "images" in batch:
        # 遍历批次中的图像
        for image_list in batch["images"]:
            for j, image in enumerate(image_list):
                image_path = dataset.list_data_dict[i]['image'][j]
                full_image_path = os.path.join(data_args.image_folder, image_path.split('###')[0])  # 获取图像的完整路径
                if not is_image_file_valid(full_image_path):
                    print(f"Image {full_image_path} is missing or invalid.")
                else:
                    print(f"Image {full_image_path} loaded successfully.")