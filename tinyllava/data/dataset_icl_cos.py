import copy
from dataclasses import dataclass
import json
from typing import Dict, Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os
import re
import matplotlib.pyplot as plt  # 添加可视化所需的库
from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *

import transformers
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


SUBIMAGE_PATTERN = r"\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]"
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with multi-image and ROI support."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    def parse_image_with_roi(self, image_info: str):
        """Helper function to extract image path and ROI from image field."""
        sub_image_info = None
        roi_str = None  # 添加这行，确保roi_str有一个默认值

        # 检查图像信息中是否包含 '###'，如果有则说明可能有ROI信息
        if '###' in image_info:
            # 直接使用 '###' 分割图像路径和ROI信息
            try:
                image_path, roi_str = image_info.split('###')
                match = re.match(SUBIMAGE_PATTERN, roi_str)
                if match:
                    coords = [float(x) for x in match.groups()]
                    sub_image_info = coords
            except (ValueError, IndexError) as e:
                print(f"Error: Failed to parse ROI from string: {roi_str}, Error: {e}")
                sub_image_info = None  # 如果解析失败，返回None
            return image_path, sub_image_info
        else:
            # 如果没有匹配到 '###'，返回图像路径和None（没有ROI信息）
            return image_info, None

    def load_image(self, image_path, roi=None):
        """Loads an image and crops it if ROI is provided."""
        image_folder = self.data_args.image_folder
        image = Image.open(os.path.join(image_folder, image_path))

        # 将图像转换为 RGB 模式，确保图像有 3 个通道
        image = image.convert('RGB')

        # 可视化原始图片
        # self.visualize_image(image, title="Original Image")

        def cropwithbbox(pil_img, sub_image_info):
            """Crops the image based on bbox coordinates [x_min, y_min, x_max, y_max]"""
            width, height = pil_img.size
            x_min, y_min, x_max, y_max = sub_image_info

            # 如果坐标都小于5，说明可能是相对坐标
            if sum([x_min, y_min, x_max, y_max]) < 5:
                # 将相对坐标转换为绝对像素坐标
                x_min = x_min * width
                y_min = y_min * height
                x_max = x_max * width
                y_max = y_max * height

            # 确定裁剪区域
            cropped_image = pil_img.crop((x_min, y_min, x_max, y_max))
            return cropped_image

        # 确保图像处理器不为 None
        if self.data_args.image_processor is None:
            raise ValueError("image_processor is not defined.")

        try:
            # 处理裁剪后的图像
            processed_image = self.image_preprocess(image)
        except Exception as e:
            # 捕获异常并输出问题的图片路径
            print(f"Error processing image: {image_path}, Error: {e}")
            return None  # 可以根据需求选择是否返回 None 或其他值

        if roi:
            # 使用新的 cropwithbbox 函数进行裁剪
            processed_image = cropwithbbox(processed_image, roi)

        # 可视化处理后的图片
        # self.visualize_image(image, title="Processed Image")

        return processed_image



    def visualize_image(self, image: Image.Image, title: str = "Image"):
        """Helper function to visualize the given image."""
        plt.figure()
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))

        images = []
        for image_info in sources['image']:
            image_path, roi = self.parse_image_with_roi(image_info)
            image = self.load_image(image_path, roi)
            if image is None:
                print(f"Image {image_path} could not be loaded.")
            images.append(image)

        if self.data_args.is_multimodal:
            if images:
                data_dict['images'] = images
            else:
                # No valid images, returning a zero tensor with the correct shape
                crop_size = getattr(self.data_args.image_processor, 'crop_size',
                                    getattr(self.data_args.image_processor, 'size'))
                data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning with multi-image support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
            if isinstance(images[0], list):
                images = torch.stack([torch.stack(img, dim=0) for img in images], dim=0)
                batch['images'] = images
            else:
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch['images'] = torch.stack(images)
                else:
                    batch['images'] = images

        return batch





def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)