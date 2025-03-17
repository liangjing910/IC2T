import faiss
import torch
import numpy as np
import json
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
print(f"Using device: {device} with {n_gpu} GPUs")

print("Loading DistilBERT tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)

print("DistilBERT tokenizer and model loaded.")


def get_text_embeddings_in_batches(texts, model, tokenizer, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # 确保输入在GPU上

        # 使用 `torch.no_grad()` 关闭梯度计算，节省内存
        with torch.no_grad():
            outputs = model(**inputs)  # DataParallel 会自动将任务分发到多个GPU
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()
            embeddings.extend(batch_embeddings)

        torch.cuda.empty_cache()

    return np.array(embeddings)


def faiss_cosine_similarity_search(embeddings, top_k=5):
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)

    index.add(normalized_embeddings.astype(np.float32))

    _, top_k_indices = index.search(normalized_embeddings.astype(np.float32), top_k)

    return top_k_indices


def group_by_dataset(data_batch):
    grouped_data = {}
    for item in data_batch:
        dataset_name = item.get('dataset')
        if dataset_name not in grouped_data:
            grouped_data[dataset_name] = []
        grouped_data[dataset_name].append(item)
    return grouped_data


def extract_main_question(question_text):
    if "<image>\n" in question_text:
        main_question = question_text.split("<image>\n")[1].split("Please provide the bounding box")[0].strip()
    else:
        main_question = question_text.strip()
    return main_question


def create_contextual_data_with_faiss(data_group, embeddings, top_k=5):
    new_data = []
    print("Processing data group for contextual similarity...")

    top_k_indices = faiss_cosine_similarity_search(embeddings, top_k=top_k)

    for i, item in enumerate(tqdm(data_group, desc="Processing data group", unit="sample")):
        most_similar_context = data_group[top_k_indices[i][1]]  # 排除第一个（即自己），选择第二个

        new_item = {
            "dataset": item["dataset"],
            "split": item["split"],
            "question_id": item["question_id"],
            "image": item["image"],
            "conversations": item["conversations"],
            "in_context_examples": [most_similar_context]
        }
        new_data.append(new_item)

    print("Contextual similarity processed.")
    return new_data


def save_to_file(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {file_name}.")


def load_large_json(file_path):
    print(f"Loading large JSON file from: {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"JSON file loaded. Total records: {len(data)}")
    return data


def process_dataset(file_path, batch_size=64, top_k=5):
    print(f"Starting dataset processing for file: {file_path}")
    data = load_large_json(file_path)
    grouped_data = group_by_dataset(data)

    for dataset_name, data_group in tqdm(grouped_data.items(), desc="Processing datasets", unit="dataset"):
        print(f"Processing dataset: {dataset_name}")

        questions = [extract_main_question(item['conversations'][0]['value']) for item in data_group]

        embeddings = get_text_embeddings_in_batches(questions, model, tokenizer, batch_size=batch_size)

        contextual_data = create_contextual_data_with_faiss(data_group, embeddings, top_k=top_k)

        save_to_file(contextual_data, f"contextual_example_output_{dataset_name}.json")

        contextual_data = []

    print("Dataset processing completed.")
    return contextual_data


file_path = 'viscot_363k.json'
print("Starting the processing pipeline...")
contextual_example = process_dataset(file_path)
print(f"Processing pipeline finished.")