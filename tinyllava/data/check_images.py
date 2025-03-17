import os
import json
from dataclasses import dataclass
from tqdm import tqdm  # 导入 tqdm 进度条库
from transformers import AutoTokenizer, AutoImageProcessor, AutoConfig
from tinyllava.data.dataset_icl import LazySupervisedDataset
from tinyllava.utils import TrainingArguments  # 确保你有这个类导入

# 定义数据集和图像路径
DATASET_PATH = "/home/LiangJing/TinyLLaVA_Factory/dataset/converted_gqa_output.json"
IMAGE_PATH = "/home/LiangJing/TinyLLaVA_Factory/dataset"

@dataclass
class DataArguments:
    data_path: str = DATASET_PATH
    image_folder: str = IMAGE_PATH  # 图像文件夹路径
    is_multimodal: bool = True
    image_processor: object = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")  # 图像处理器
    conv_version: str = "phi"  # 对话版本号
    image_aspect_ratio: str = "pad"  # 图像纵横比设置

# 假设 TrainingArguments 中包含模型路径等信息
training_arguments = TrainingArguments(
    pretrained_model_path="/home/LiangJing/TinyLLaVA_Factory/TinyLLaVA-Phi-2-SigLIP-3.1B",  # 在这里指定你的预训练模型路径
    output_dir="/home/LiangJing/TinyLLaVA_Factory/output"  # 指定训练输出的目录
)

config = AutoConfig.from_pretrained(
        training_arguments.pretrained_model_path,
        model_type="tinyllava",  # 使用你注册的自定义模型类型
        local_files_only=True
    )
tokenizer = AutoTokenizer.from_pretrained(training_arguments.pretrained_model_path, use_fast=False,
                                          model_max_length=config.tokenizer_model_max_length,
                                          padding_side=config.tokenizer_padding_side)

data_args = DataArguments()

# 创建 LazySupervisedDataset 实例
dataset = LazySupervisedDataset(data_path=DATASET_PATH, tokenizer=tokenizer, data_args=data_args)


# 测试方法，解析和加载图像
def test_image_loading(dataset):
    """
    测试 parse_image_with_roi 和 load_image 方法.
    """
    for i in tqdm(range(len(dataset)), desc="Testing images", unit="record"):
        record = dataset.list_data_dict[i]
        images = record.get("image", [])

        for image_info in images:
            # 直接调用已经写好的 parse_image_with_roi 和 load_image 方法
            image_path, roi = dataset.parse_image_with_roi(image_info)

            # 确保解析出了图像路径
            if image_path:
                full_image_path = os.path.join(IMAGE_PATH, image_path)
                # 检查图片是否存在
                if os.path.exists(full_image_path):
                    try:
                        # 加载并处理图像
                        image = dataset.load_image(image_path, roi)
                        if image is None:
                            print(f"Failed to process image: {image_path}")
                    except Exception as e:
                        print(f"Error processing image: {full_image_path}, ROI: {roi}, Error: {e}")
                else:
                    print(f"Image file does not exist: {full_image_path}")
            else:
                print(f"Failed to parse image path from: {image_info}")


# 执行测试
test_image_loading(dataset)