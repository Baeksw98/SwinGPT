
import torch
from swingpt.utils.train_utils import *
from swingpt.evaluation.swingpt_evaluator import * 
from swingpt.model.model_loader import *

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

def evaluate(config_dir, debug_mode=False):
    # Load arguments from config file
    model_args, data_args, training_args = load_configuration(config_dir)

    # Instantiate model and tokenizer 
    model, tokenizer = load_model_and_tokenizer(model_args=model_args, training_args=training_args, data_args=data_args)

    # Define the default conversation based on model arg version
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]

    # Instantiate the Evaluator class
    SG_IC_Evaluator = SwinGPTEvaluator(
        model=model, tokenizer=tokenizer, training_args=training_args, mode="IC"
    )
    SG_OD_Evaluator = SwinGPTEvaluator(
        model=model, tokenizer=tokenizer, training_args=training_args, mode="OD"
    )
    
    # Set up WandbLogger
    wandb_logger = WandbLogger(name='SwinGPTModel', project='SwinGPT_Project', log_model='all')
    
    # Instantiate DDP Strategy
    ddp_strategy = DDPStrategy(find_unused_parameters=True)
    
    if debug_mode: 
        # Instantiate the trainer with configurations for single-GPU or CPU
        trainer = Trainer(
            devices=1 if torch.cuda.is_available() else 0,
            precision='16-mixed' if training_args.fp16 else '32-true', 
            enable_checkpointing=False,  
            enable_progress_bar=True,   
            limit_test_batches=2,  
        )
    else:     
        # Instantiate the trainer class (DDP, 2 nodes, 4 gpus per each node)
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=training_args.num_train_epochs,
            num_nodes=2, # 2 GPU servers
            devices=4,  # 4 GPUs per each server
            accelerator='gpu' if torch.cuda.is_available() else None,  
            strategy='ddp',
            precision='16-mixed' if training_args.fp16 else '32-true',  
            enable_checkpointing=False,  
            enable_progress_bar=True,  
            limit_test_batches=0.05,  
            log_every_n_steps=training_args.logging_steps,
        )

    # Prepare data
    eval_data_module = EvaluatorDataModule(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    eval_data_module.setup()

    # Create OD and IC dataloaders
    IC_dataloader = eval_data_module.test_dataloader('IC')
    OD_dataloader = eval_data_module.test_dataloader('OD')

    # Run evaluation for Image Captioning dataset
    trainer.test(SG_IC_Evaluator, dataloaders=IC_dataloader)

    # Run evaluation for Object Detection dataset 
    trainer.test(SG_OD_Evaluator, dataloaders=OD_dataloader)    

    print("Evaluation complete")
    
if __name__ == "__main__":
    # Load config directory
    config_dir = "/data/cad-recruit-02_814/swbaek/config/config_arguments_test.yaml"
    
    # Run evaluation 
    evaluate(config_dir, debug_mode=False)
