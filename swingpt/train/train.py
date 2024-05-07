import transformers
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

from swingpt.utils import conversation as conversation_lib
from swingpt.utils.constants import *
from swingpt.utils.train_utils import *
from swingpt.datasets.dataset_processor import *
from swingpt.train.swingpt_trainer import *
from swingpt.model.SwinGPT import SwinGPTForCausalLM

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

def train(config_dir, debug_mode=False):
    if debug_mode: 
        num_nodes = 1
        devices = 1
    else:
        num_nodes = 2
        devices = 4
        # Set up DDP environment 
        setup_dist()

    # Load data, train, and model arguments    
    model_args, data_args, training_args = load_configuration(config_dir)

    # Initialize additional_configs dictionary
    additional_configs = {}

    # Set up for model quantization configurations if specified in training_args
    if training_args.bits in [4, 8]:
        additional_configs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(torch.float16 if training_args.fp16 else torch.float32),
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type
        )
        # Use low CPU memory usage setting
        additional_configs['low_cpu_mem_usage'] = True
    
    # Instantiate model
    model = SwinGPTForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        cache_dir=training_args.cache_dir, 
        use_cache=False,
        **additional_configs
        )

    # Freezing backbone architecture
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # Keeping track of gradient checkpoints
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # When enabled quantization get model from kbit training
    if training_args.bits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

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

    # Instantiate the Tokenizer (GPT2-base)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        cache_dir=training_args.cache_dir, 
        model_max_length=training_args.model_max_length, 
        padding_side="right", 
        use_fast=False,)

    # Get pad token and Define default conversation
    smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token="[PAD]"), 
                                        tokenizer=tokenizer, model=model,)
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    
    if model_args.vision_tower is not None:
        # Initialize vision tower 
        model.get_model().initialize_vision_modules(model_args=model_args)
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.float16, device=training_args.device)

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

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

    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=training_args.output_dir,  
        filename='{epoch}-{step}',  
        save_top_k=1, 
        verbose=True,  
        monitor='train_loss',  
        mode='min', 
        save_last=True,  
        every_n_train_steps=training_args.save_steps,  
        auto_insert_metric_name=False,
    )

    # setup model, tokenizer, and args
    model_module = SwinGPTTrainer(model, tokenizer, training_args)
    data_module = TrainerDataModule(tokenizer, data_args, training_args)
    
    # Get latest checkpoint of model training
    latest_checkpoint = get_latest_checkpoint(training_args.output_dir)
    if latest_checkpoint:
        print(f"Resuming training from {latest_checkpoint}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Set up WandbLogger
    wandb_logger = WandbLogger(name='SwinGPTModel', project='SwinGPT_Project', log_model='all')
    
    # Instantiate DDP Strategy
    ddp_strategy = DDPStrategy(find_unused_parameters=True)
    
    # Instantiate the trainer class (DDP, 2 nodes, 4 gpus per each node)
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=training_args.num_train_epochs,
        num_nodes=num_nodes, 
        devices=devices,  
        accelerator='gpu' if torch.cuda.is_available() else None,  
        strategy=ddp_strategy,  
        callbacks=[checkpoint_callback],
        precision='16-mixed' if training_args.fp16 else '32-true',  
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        log_every_n_steps=64,
        val_check_interval=0.1,
        limit_val_batches=0.1,        
    )
    
    # Start training and potentially resume from the latest checkpoint
    trainer.fit(model_module, datamodule=data_module, ckpt_path=latest_checkpoint)

if __name__ == "__main__":
    # Define config directory 
    config_dir = "/data/cad-recruit-02_814/swbaek/config/config_arguments_train.yaml"
    
    # Run train
    train(config_dir, debug_mode=False)