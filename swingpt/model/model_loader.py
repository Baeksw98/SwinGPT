import torch
from transformers import AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

from swingpt.model.SwinGPT import SwinGPTForCausalLM
from swingpt.utils.train_utils import *

def load_model_and_tokenizer(model_args, training_args, data_args):
    # Instantiate the Tokenizer (GPT2-base)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        cache_dir=training_args.cache_dir, 
        model_max_length=training_args.model_max_length, 
        padding_side="right", 
        use_fast=False,)
    
    # Load model from checkpoint
    model = SwinGPTForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        cache_dir=training_args.cache_dir, 
        use_cache=True)

    # Set up LoRA configurations if enabled
    if training_args.lora_enable:
        target_modules = ["c_attn", "c_proj", "c_fc"]
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            fan_in_fan_out=True,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        print("LoRA adapters are added")
        model.print_trainable_parameters()
        
    model.get_model().initialize_vision_modules(model_args=model_args)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.float16, device=training_args.device)
    
    # Set necesseary updates to config and arguments
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_use_images = data_args.mm_use_images = model_args.mm_use_images
    training_args.mm_use_images = model_args.mm_use_images
    data_args.image_processor = vision_tower.image_processor

    # Initialize vision tokenizer and save to model 
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    
    # Get pad token
    smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token="[PAD]"), 
                                        tokenizer=tokenizer, model=model,)    
    
    # Load checkpoint
    checkpoint = torch.load(model_args.checkpoint_path)
    state_dict = checkpoint['state_dict']

    # Remove 'model.' prefix from state_dict keys if present
    new_state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    # Prepare model for evaluation/testing
    model.eval()

    return model, tokenizer