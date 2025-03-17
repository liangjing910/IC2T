import argparse

import numpy as np
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from transformers.image_transforms import to_pil_image

from tinyllava.model.load_model_icl import load_pretrained_model
from tinyllava.utils import *
from tinyllava.data import *

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, text_processor, image_processor):
        self.questions = questions
        self.image_folder = image_folder
        self.text_processor = text_processor
        self.image_processor = image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        question_id = line["question_id"]

        # Load main image
        main_image_file = line["image"]
        main_image = Image.open(os.path.join(self.image_folder, main_image_file)).convert('RGB')
        main_image_tensor = self.image_processor(main_image)

        # Load in-context example image and data
        context_image_file, bbox_str = line["in_context_example"]["image"].split("###")
        context_image = Image.open(os.path.join(self.image_folder, context_image_file)).convert('RGB')
        context_image_tensor = self.image_processor(context_image)

        # Concatenate images along a new dimension to form [img_num, channels, height, width]
        images_tensor = torch.stack([context_image_tensor, main_image_tensor], dim=0)

        context_text = line["in_context_example"]["text"]
        context_answer = line["in_context_example"]["answer"]

        # Main question text with additional prompt for bbox
        main_text = line["text"]
        first_round_prompt = (
                DEFAULT_IMAGE_TOKEN + '\n' + context_text + "\nThe bounding box coordinate of the region that can help you answer the question better is " + bbox_str + '\n' +
                DEFAULT_IMAGE_TOKEN + '\n' + main_text + "\nPlease provide the bounding box coordinate of the region that can help you answer the question better."
        )


        msg_first = Message()
        msg_first.add_message(first_round_prompt)
        result_first = self.text_processor(msg_first.messages, mode='eval')
        input_ids_first = result_first['input_ids']

        # Return images_tensor as [img_num, channels, height, width]
        return question_id, input_ids_first, images_tensor, main_text, np.array(main_image), context_answer

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    question_ids, input_ids_first, images_tensors, main_texts, main_images = zip(*batch)
    input_ids_first = torch.stack(input_ids_first, dim=0)
    images_tensors = torch.stack(images_tensors, dim=0)
    return question_ids, input_ids_first, images_tensors, main_texts, main_images

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

# DataLoader
def create_data_loader(questions, image_folder, text_processor, image_processor, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, text_processor, image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                             collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, text_processor, image_processor)
    model.to(device='cuda')

    for question_ids, input_ids_first, images_tensor, main_text, main_image, context_answer in tqdm(data_loader, total=len(questions)):
        idx = question_ids[0]

        # Move input tensors to CUDA
        input_ids_first = input_ids_first.to(device='cuda', non_blocking=True)
        images_tensor = images_tensor.to(device='cuda', dtype=torch.float16, non_blocking=True)
        attention_mask = torch.ones(input_ids_first.shape, dtype=torch.long, device='cuda')
        # First round to obtain bbox coordinate
        with torch.inference_mode():
            output_ids_first = model.generate(
                input_ids_first,
                attention_mask=attention_mask,
                images=images_tensor,  # images_tensor is now [batch_size, img_num, channels, height, width]
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )

        bbox_coordinates = tokenizer.batch_decode(output_ids_first, skip_special_tokens=True)[0].strip()
        main_image = Image.fromarray(np.array(main_image[0])).convert('RGB')
        try:
            coords = list(map(float, bbox_coordinates.strip('[]').split(',')))
            if len(coords) == 4:
                cropped_image_pil = cropwithbbox(main_image, coords)  # 使用 cropwithbbox 裁剪

                # 使用 image_processor 处理裁剪后的图像
                cropped_image = image_processor(cropped_image_pil)
            else:
                print(f"Invalid bbox format for question {idx}. Skipping second round.")
                cropped_image = image_processor(main_image)
        except ValueError as e:
            print(f"Error in cropping for question {idx}: {e},bbox_coordinates ={bbox_coordinates},coords ={coords}")
            cropped_image = image_processor(main_image)
        # Second round to answer the final question based on cropped image
        second_round_prompt = (
                DEFAULT_IMAGE_TOKEN + '\n' + "This image is the original image.\n" +
                DEFAULT_IMAGE_TOKEN + '\n' + "This image is the local detail image.\n" +
                " According to the original image and local detail image analysis, the answer is "+ context_answer +
                DEFAULT_IMAGE_TOKEN + '\n' + "This image is the original image.\n" +
                DEFAULT_IMAGE_TOKEN + '\n' + "This image is the local detail image.\n" +
                "Please answer the question based on the original image and local detail image."
        )

        msg_second = Message()
        msg_second.add_message(second_round_prompt)
        result_second = text_processor(msg_second.messages, mode='eval')
        input_ids_second = result_second['input_ids'].to(device='cuda', non_blocking=True)
        # 增加批次维度
        input_ids_second = input_ids_second.unsqueeze(0)  # 调整为 [1, sequence_length]
        attention_mask = torch.ones(input_ids_second.shape, dtype=torch.long, device='cuda')
        with torch.inference_mode():
            output_ids_second = model.generate(
                input_ids_second,
                attention_mask=attention_mask,
                images=cropped_image.unsqueeze(0).unsqueeze(0).to(dtype=torch.float16, device='cuda',
                                                                  non_blocking=True),
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )

        # Decode final answer
        final_answer = tokenizer.batch_decode(output_ids_second, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "bbox_coordinates": bbox_coordinates,
                                   "text": final_answer,
                                   "answer_id": ans_id,
                                   "model_id": args.model_base,
                                   "metadata": {}}) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)