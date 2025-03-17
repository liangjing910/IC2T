import tokenizers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor

from tinyllava.train.tinyllava_trainer import LLaVATrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset_icl_cos import make_supervised_data_module
from tinyllava.model.modeling_tinyllava_phi import TinyLlavaConfig, TinyLlavaForConditionalGeneration  # 导入自定义的模型类和配置类


# 注册自定义配置类和模型类
AutoConfig.register("tinyllava", TinyLlavaConfig)
AutoModelForCausalLM.register(TinyLlavaConfig, TinyLlavaForConditionalGeneration)

def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio





def train():

    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    logger_setting(getattr(training_arguments, 'output_dir', None))
    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    load_settings(model_arguments, data_arguments, training_arguments)
    # load pretrained checkpoint
    config = AutoConfig.from_pretrained(
        training_arguments.pretrained_model_path,
        model_type="tinyllava",  # 使用你注册的自定义模型类型
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        training_arguments.pretrained_model_path,
        config=config,  # 使用已加载的配置
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(training_arguments.pretrained_model_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
    model.tokenizer = tokenizer
    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    data_arguments.image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name_or_path)
    data_arguments.is_multimodal = True
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3
    trainer = LLaVATrainer(model=model, #does not require model.to(device), huggingface/deepspeed does it for you?
                           tokenizer=tokenizer,
                           args=training_arguments,
                           **data_module)
    
    trainer.train()
    
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
