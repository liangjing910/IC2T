import json
from pathlib import Path

# 基础路径
BASE_PATH = "/home/LiangJing"

# 推理结果文件路径
predictions_path = Path(BASE_PATH) / "TinyLLaVA_Factory/dataset/eval/gqa/results.json"

# 推理验证数据集的路径
validation_data_path = Path(BASE_PATH) / "TinyLLaVA_Factory/dataset/eval/gqa/converted_gqa_output_eval.json"

# 加载推理结果和验证数据集
with open(predictions_path, "r") as f:
    predictions = json.load(f)

with open(validation_data_path, "r") as f:
    validation_data = json.load(f)

# 将验证数据中的真实答案提取出来，以 question_id 为键，便于快速查找
ground_truth_dict = {}
for item in validation_data:
    question_id = item["question_id"]
    # 提取 `conversations` 中最后一个 gpt 的 answer 作为真实答案
    ground_truth_answer = item["conversations"][-1]["value"]
    ground_truth_dict[question_id] = ground_truth_answer

# 计算准确度
correct = 0
total = len(predictions)

for prediction in predictions:
    question_id = prediction["question_id"]
    predicted_answer = prediction["second_round_answer"]

    # 获取真实答案并比较
    if question_id in ground_truth_dict:
        true_answer = ground_truth_dict[question_id]
        if predicted_answer == true_answer:
            correct += 1

accuracy = correct / total * 100  # 转为百分比形式

print(f"Accuracy: {accuracy:.2f}%")