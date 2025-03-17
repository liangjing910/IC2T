import subprocess
import os
import sys


def run_custom_finetune():
    # 设置环境变量
    data_path = "/data/vipuser/TinyLLaVA_Factory/dataset/converted_gqa_output_v7.json"
    image_path = "/data/vipuser/TinyLLaVA_Factory/dataset"
    model_max_length = 3072
    output_dir = "/data/vipuser/TinyLLaVA_Factory/checkpoints/llava_factory/custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora-icl"

    # 检测是否在调试模式下
    def is_debugging():
        gettrace = getattr(sys, 'gettrace', None)
        return gettrace is not None and gettrace()

    # 构建命令行
    if is_debugging():
        print("Debug mode detected, disabling distributed training.")
        command = [
            "/home/vipuser/miniconda3/envs/tinyllava_factory/bin/python3.10",  # 不使用 deepspeed，直接调用 python
            "custom_finetune_icl.py",
            "--local_rank", "-1",
            "--data_path", data_path,
            "--image_folder", image_path,
            "--is_multimodal", "True",
            "--conv_version", "phi",
            "--mm_vision_select_layer", "-2",
            "--image_aspect_ratio", "pad",
            "--fp16", "True",
            "--training_recipe", "lora",
            "--tune_type_llm", "lora",
            "--tune_type_vision_tower", "frozen",
            "--tune_vision_tower_from_layer", "0",
            "--tune_type_connector", "full",
            "--lora_r", "128",
            "--lora_alpha", "256",
            "--group_by_modality_length", "False",
            "--pretrained_model_path", "/data/vipuser/TinyLLaVA_Factory/TinyLLaVA-Phi-2-SigLIP-3.1B",
            "--output_dir", output_dir,
            "--num_train_epochs", "1",
            "--per_device_train_batch_size", "2",
            "--per_device_eval_batch_size", "4",
            "--gradient_accumulation_steps", "16",
            "--evaluation_strategy", "no",
            "--save_strategy", "steps",
            "--save_steps", "50000",
            "--save_total_limit", "1",
            "--learning_rate", "1e-4",
            "--weight_decay", "0.",
            "--warmup_ratio", "0.03",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "1",
            "--tf32", "False",
            "--model_max_length", str(model_max_length),
            "--gradient_checkpointing", "True",
            "--dataloader_num_workers", "8",
            "--lazy_preprocess", "True",
            "--report_to", "tensorboard",
            "--tokenizer_use_fast", "False",
            "--run_name", "custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora-icl"
        ]
    else:
        # 正常运行分布式
        command = [
            "/home/LiangJing/miniconda3/envs/tinyllava_factory/bin/deepspeed",  # 完整路径
            "--include", "localhost:0,1,2,3", "--master_port", "29501",
            "custom_finetune_icl.py",
            "--deepspeed", "/home/LiangJing/TinyLLaVA_Factory/scripts/zero2.json",
            "--data_path", data_path,
            "--image_folder", image_path,
            "--is_multimodal", "True",
            "--conv_version", "phi",
            "--mm_vision_select_layer", "-2",
            "--image_aspect_ratio", "pad",
            "--fp16", "True",
            "--training_recipe", "lora",
            "--tune_type_llm", "lora",
            "--tune_type_vision_tower", "frozen",
            "--tune_vision_tower_from_layer", "0",
            "--tune_type_connector", "full",
            "--lora_r", "128",
            "--lora_alpha", "256",
            "--group_by_modality_length", "False",
            "--pretrained_model_path", "/home/LiangJing/TinyLLaVA_Factory/TinyLLaVA-Phi-2-SigLIP-3.1B",
            "--output_dir", output_dir,
            "--num_train_epochs", "1",
            "--per_device_train_batch_size", "2",
            "--per_device_eval_batch_size", "4",
            "--gradient_accumulation_steps", "16",
            "--evaluation_strategy", "no",
            "--save_strategy", "steps",
            "--save_steps", "50000",
            "--save_total_limit", "1",
            "--learning_rate", "1e-4",
            "--weight_decay", "0.",
            "--warmup_ratio", "0.03",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "1",
            "--tf32", "False",
            "--model_max_length", str(model_max_length),
            "--gradient_checkpointing", "True",
            "--dataloader_num_workers", "8",
            "--lazy_preprocess", "True",
            "--report_to", "tensorboard",
            "--tokenizer_use_fast", "False",
            "--run_name", "custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora-icl"
        ]

    # 执行命令
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e.stderr}")


if __name__ == "__main__":
    run_custom_finetune()