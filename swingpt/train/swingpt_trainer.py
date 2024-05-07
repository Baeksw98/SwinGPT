from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda import OutOfMemoryError

from swingpt.datasets.dataset_processor import *
from swingpt.utils.train_utils import * 

from pytorch_lightning import LightningDataModule, LightningModule

class SwinGPTTrainer(LightningModule):
    def __init__(self, model, tokenizer, training_args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.save_hyperparameters('training_args')
        
        # Store batch sizes
        self.train_batch_size = training_args.per_device_train_batch_size
        self.eval_batch_size = training_args.per_device_eval_batch_size
        
    def forward(self, input_ids, attention_mask=None, labels=None, images=None, bbox_inputs=None, bbox_labels=None):
        # Pass all inputs to the model's forward method. Match these parameters with those expected by your model.
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            images=images, bbox_inputs=bbox_inputs, bbox_labels=bbox_labels, return_dict=True)
        return outputs
    
    def log_metrics(self, loss, prefix="train"):
        # Determine the batch size based on the prefix
        batch_size = self.train_batch_size if prefix == "train" else self.eval_batch_size

        # Log total loss
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, 
                logger=True, sync_dist=True, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        try:
            outputs = self(**batch)
            train_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
            self.log_metrics(train_loss, prefix="train")
        
        except OutOfMemoryError:
            torch.cuda.empty_cache()  
            print(f"Ran out of memory on step {batch_idx}, cache cleared. Retrying step...")
            outputs = self(**batch)  # Retry the current batch
            train_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
            self.log_metrics(train_loss, prefix="train")

        # Clear cache periodically every 500 steps
        if batch_idx % 500 == 0:
            torch.cuda.empty_cache()
            print("Cleared CUDA cache at step:", batch_idx)

        return train_loss

    def validation_step(self, batch, batch_idx):
        try:
            outputs = self(**batch)
            val_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
            self.log_metrics(val_loss, prefix="val")
        
        except OutOfMemoryError:
            torch.cuda.empty_cache()  
            print(f"Ran out of memory during validation on step {batch_idx}, cache cleared. Retrying step...")
            outputs = self(**batch)  # Retry the current validation batch
            val_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
            self.log_metrics(val_loss, prefix="val")

        # Clear cache periodically every 500 steps during validation too
        if batch_idx % 500 == 0:
            torch.cuda.empty_cache()
            print("Cleared CUDA cache at validation step:", batch_idx)
            
        return val_loss
    
    def configure_optimizers(self):
        # Define different parameter groups with specific optimization configs
        decay_param_names = set(get_parameter_names(self.model, ['LayerNorm', 'bias']))  # Get names of parameters
        projector_param_names = set(n for n, p in self.model.named_parameters() if "mm_projector" in n) if self.training_args.mm_projector_lr is not None else set()
        
        # Exclude projector parameters from base and decay parameters if they have a special learning rate
        base_parameters = [p for n, p in self.model.named_parameters() if n not in decay_param_names and n not in projector_param_names]
        decay_parameters = [p for n, p in self.model.named_parameters() if n in decay_param_names and n not in projector_param_names]
        
        optimizer_grouped_parameters = [
            {'params': base_parameters, 'weight_decay': 0.0},  # Base parameters have no weight decay
            {'params': decay_parameters, 'weight_decay': self.training_args.weight_decay}  # Decay parameters have weight decay
        ]

        # Special handling for mm_projector parameters if a separate learning rate is specified
        if self.training_args.mm_projector_lr is not None:
            projector_parameters = [p for n, p in self.model.named_parameters() if n in projector_param_names]
            optimizer_grouped_parameters.append({
                'params': projector_parameters,
                'lr': self.training_args.mm_projector_lr,  # Special learning rate for projector parameters
                'weight_decay': self.training_args.weight_decay
            })

        # Create optimizer with grouped parameters
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_args.learning_rate)

        # Setup scheduler if specified
        scheduler = None
        if self.training_args.lr_scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.training_args.num_train_epochs, eta_min=0)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}
        
        return optimizer
    
class TrainerDataModule(LightningDataModule):
    def __init__(self, tokenizer, data_args, training_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.batch_size_train = training_args.per_device_train_batch_size
        self.batch_size_val = training_args.per_device_eval_batch_size
        self.data_loader_workers = training_args.dataloader_num_workers

    def setup(self, stage=None):
        # Called on each GPU separately
        self.train_dataset = LazyDataset(
            json_file_path=self.data_args.json_file_path_train,
            tokenizer=self.tokenizer,
            data_args=self.data_args
        )
        self.val_dataset = LazyDataset(
            json_file_path=self.data_args.json_file_path_val,
            tokenizer=self.tokenizer,
            data_args=self.data_args
        )
        self.data_collator = DataCollator(tokenizer=self.tokenizer, is_eval_mode=self.data_args.is_eval_mode)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True,
                        collate_fn=self.data_collator, num_workers=self.data_loader_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_val, shuffle=False,
                        collate_fn=self.data_collator, num_workers=self.data_loader_workers, pin_memory=True)

