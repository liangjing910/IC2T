import json

DATASET_PATH = "/home/LiangJing/TinyLLaVA_Factory/dataset/converted_gqa_output.json"

# 读取数据集
with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

# 过滤掉包含指定图片名的样本
filtered_data = [entry for entry in data if not any(img for img in entry['image'] if '2372349.jpg' in img or '2379318.jpg' in img)]

# 将过滤后的数据重新写回文件
with open(DATASET_PATH, 'w') as f:
    json.dump(filtered_data, f, indent=4)

print(f"已删除包含 '2372349.jpg' 或 '2379318.jpg' 图片的样本，更新后的数据已保存。")