import json
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from PIL import Image
from pathlib import Path
from tqdm import tqdm  # 导入 tqdm 用于显示进度条

from tinyllava.data import ImagePreprocess, TextPreprocess
from tinyllava.model.load_model_icl import load_pretrained_model
from tinyllava.utils import *

# 基础路径
BASE_PATH = "/home/LiangJing"
IMAGE_PATH = Path(BASE_PATH) / "TinyLLaVA_Factory/dataset"
MODEL_PATH = Path(BASE_PATH) / "TinyLLaVA_Factory/checkpoints/llava_factory/custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora-icl"

# 加载模型和处理器
model, tokenizer, image_processor, context_len = load_pretrained_model(str(MODEL_PATH))

# 确保模型在 GPU 上
model = model.to("cuda")

# 配置文本和图像处理器
args = {"conv_mode": "phi"}  # 假设使用的对话模式为 "phi"
text_processor = TextPreprocess(tokenizer, args["conv_mode"])
data_args = model.config
image_processor = ImagePreprocess(image_processor, data_args)

# 获取停止符号
stop_str = text_processor.template.separator.apply()[1]

# 加载数据
data_path = IMAGE_PATH / "eval/gqa/converted_gqa_output_eval.json"
with open(data_path, "r") as f:
    data = json.load(f)

results = []

# 推理参数设置
temperature = 0
top_p = None
num_beams = 1
max_new_tokens = 128  # 根据需求设置最大生成长度

# 使用 tqdm 进行进度条显示
for item in tqdm(data, desc="Processing Items"):
    sample_result = {
        "dataset": item["dataset"],
        "split": item["split"],
        "question_id": item["question_id"],
        "first_round_bbox": None,
        "second_round_answer": None
    }

    # 第一轮推理
    first_question = item["conversations"][0]["value"]
    first_image_path = IMAGE_PATH / item["image"][0]
    second_image_path = IMAGE_PATH / item["image"][1]

    # 加载并处理图像
    first_image = image_processor(Image.open(first_image_path))
    second_image = image_processor(Image.open(second_image_path))

    # 将图像堆叠后传入模型，并确保图像在 GPU 上
    stacked_images_first_round = torch.stack([first_image, second_image], dim=0).cuda()

    # 准备输入并进行第一轮推理
    prompt = first_question.replace("<image>", DEFAULT_IMAGE_TOKEN, 1).replace("<image>", DEFAULT_IMAGE_TOKEN, 1)
    msg = Message()
    msg.add_message(prompt)
    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids'].unsqueeze(0).cuda()  # 确保 input_ids 在 GPU 上
    # 加载 tokenizer 后检查 pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 将 pad_token_id 设置为 eos_token_id

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()  # 创建 attention mask

    # 停止条件设置
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=stacked_images_first_round,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)].strip()
    sample_result["first_round_bbox"] = outputs  # 假设输出为坐标字符串

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

    # 第二轮推理
    second_question = item["conversations"][2]["value"]

    # 加载并处理第二轮图像
    # 第一张图片：使用 image 列表中的第三张图片
    third_image_path = IMAGE_PATH / item["image"][2]
    third_image = image_processor(Image.open(third_image_path))

    # 第二张图片：使用 image 列表中的第四张图片，并按 `###` 后面的坐标进行裁剪
    fourth_image_path_full = item["image"][3]
    if '###' in fourth_image_path_full:
        fourth_image_path, roi_coords_str = fourth_image_path_full.split("###")
        roi_coords_1 = [float(x) for x in roi_coords_str[1:-1].split(",")]
    else:
        fourth_image_path = fourth_image_path_full
        roi_coords_1 = None

    fourth_image_path = IMAGE_PATH / fourth_image_path
    fourth_image = Image.open(fourth_image_path)
    if roi_coords_1:
        fourth_image_cropped_1 = image_processor(cropwithbbox(fourth_image, roi_coords_1))
    else:
        fourth_image_cropped_1 = image_processor(fourth_image)

    # 第三张图片：使用 image 列表中的第五张图片
    fifth_image_path = IMAGE_PATH / item["image"][4]
    fifth_image = image_processor(Image.open(fifth_image_path))

    # 第四张图片：使用第五张图片，并基于第一轮推理的 bbox 坐标进行裁剪
    # 假设第一轮推理的 bbox 是 "[x_min, y_min, x_max, y_max]" 格式
    roi_coords_2 = [float(x) for x in outputs[1:-1].split(",")]  # 从第一轮推理输出中解析坐标
    fifth_image_cropped_2 = image_processor(cropwithbbox(Image.open(fifth_image_path), roi_coords_2))

    # 将第二轮图像堆叠后传入模型
    stacked_images_second_round = torch.stack(
        [third_image, fourth_image_cropped_1, fifth_image, fifth_image_cropped_2], dim=0
    ).cuda()

    # 准备输入并进行第二轮推理
    prompt = second_question.replace("<image>", DEFAULT_IMAGE_TOKEN, 4)
    msg = Message()
    msg.add_message(prompt)
    processed_text = text_processor(msg.messages, mode='eval')
    input_ids = processed_text['input_ids'].unsqueeze(0).cuda()  # 确保 input_ids 在 GPU 上
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()  # 创建 attention mask
    # 停止条件设置
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=stacked_images_second_round,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)].strip()
    sample_result["second_round_answer"] = outputs

    results.append(sample_result)

# 保存结果到 JSON 文件
output_path = IMAGE_PATH / "eval/gqa/results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("推理完成，结果已保存到", output_path)