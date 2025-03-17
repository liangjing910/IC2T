import json
import os
import re
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
from dataclasses import dataclass
from tinyllava.data.dataset_icl import LazySupervisedDataset, ImagePreprocess
from tqdm import tqdm

# 定义数据集和图像路径
DATASET_PATH = "/data/vipuser/TinyLLaVA_Factory/dataset/converted_gqa_output_v7.json"
IMAGE_PATH = "/data/vipuser/TinyLLaVA_Factory/dataset"
SUBIMAGE_PATTERN = r"\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]"

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


# 初始化图像处理器和 ImagePreprocess
data_args = DataArguments()
image_preprocessor = ImagePreprocess(image_processor=data_args.image_processor, data_args=data_args)


# 定义解析 ROI 的函数
def parse_image_with_roi(image_info: str):
    """Helper function to extract image path and ROI from image field."""
    sub_image_info = None
    roi_str = None  # 确保 roi_str 有默认值

    # 检查图像信息中是否包含 '###'，如果有则说明可能有 ROI 信息
    if '###' in image_info:
        try:
            image_path, roi_str = image_info.split('###')
            match = re.match(SUBIMAGE_PATTERN, roi_str)
            if match:
                coords = [float(x) for x in match.groups()]
                sub_image_info = coords
        except (ValueError, IndexError) as e:
            print(f"Error: Failed to parse ROI from string: {roi_str}, Error: {e}")
            sub_image_info = None  # 如果解析失败，返回 None
        return image_path, sub_image_info
    else:
        return image_info, None


# 定义一个加载并裁剪图像的函数
def load_image(image_path, roi=None):
    """Loads an image and crops it if ROI is provided."""
    try:
        image = Image.open(image_path).convert('RGB')
    except (IOError, SyntaxError) as e:
        print(f"Invalid image file at {image_path}: {e}")
        return None

    if roi:
        width, height = image.size
        x_min, y_min, x_max, y_max = roi
        if sum([x_min, y_min, x_max, y_max]) < 5:
            x_min = x_min * width
            y_min = y_min * height
            x_max = x_max * width
            y_max = y_max * height
        image = image.crop((x_min, y_min, x_max, y_max))

    return image


# 使用图像处理器处理图像
def process_image_with_preprocessor(image, image_path):
    try:
        processed_image = image_preprocessor(image)
        return processed_image
    except Exception as e:
        print(f"Error processing image at {image_path}: {e}")
        return None


# 遍历数据集，验证每个图像路径、格式并裁剪图像
for entry in tqdm(dataset_data, desc="Checking dataset"):
    for image_info in entry["image"]:
        image_path, roi = parse_image_with_roi(image_info)
        full_image_path = os.path.join(data_args.image_folder, image_path)

        # 加载并裁剪图像
        cropped_image = load_image(full_image_path, roi)
        if cropped_image is None:
            print(f"Image {full_image_path} is missing or invalid.")
            continue

        # 使用图像处理器处理裁剪后的图像，并传入图片路径
        processed_image = process_image_with_preprocessor(cropped_image, full_image_path)
        if processed_image is None:
            print(f"Image {full_image_path} could not be processed by the image processor.")


print("Dataset check completed.")