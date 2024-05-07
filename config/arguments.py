from dataclasses import dataclass, field
from typing import Optional
import transformers

@dataclass
class DataArguments:
    # Paths to datasets
    json_file_path_train: str = field(default=None, metadata={"help": "Path to the training data."})
    json_file_path_val: str = field(default=None, metadata={"help": "Path to the validation data."})
    test_IC_path: str = field(default=None, metadata={"help": "Path to the test IC data."})
    test_OD_path: str = field(default=None, metadata={"help": "Path to the test OD data."})

    # Data processing configurations
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'pad'  
    is_eval_mode: bool = False

@dataclass
class ModelArguments:
    # Basic model configurations
    model_name_or_path: Optional[str] = field(default="None")
    version: Optional[str] = field(default="v0")
    checkpoint_path: str = field(default=None, metadata={"help": "Path to the checkpoint of trained model."})

    # Model components and their adjustments
    freeze_backbone: bool = False
    tune_mm_mlp_adapter: bool = False
    vision_tower: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrained_mm_projector: Optional[str] = field(default=None)

    # Multimodal settings
    mm_use_images: bool = True
    mm_use_bboxes: bool = True
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_projector_type: Optional[str] = field(default='linear')
    mm_patch_merge_type: Optional[str] = field(default='flat')

    # Attention settings
    add_cross_attention: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Directories
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None)

    # Model & Training Settings
    optim: str = field(default="adamw_torch")
    strategy: str = field(default="ddp")
    model_max_length: int = field(default=1024, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    local_rank: int = None

    # Data Type & Quantization
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)  # Not supported in V100 GPUs
    tf32: bool = field(default=False)  # Not supported in V100 GPUs
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})

    # Multi-modal and LoRA Settings
    freeze_mm_mlp_adapter: bool = field(default=False)
    mm_projector_lr: Optional[float] = None
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")

    # Optimization & Performance
    gradient_checkpointing: bool = field(default=False)
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": True})

    # Beam Search and Sampling
    temperature: float = field(default=0.7)
    max_new_tokens: int = field(default=50)
    num_beams: int = field(default=20)
    no_repeat_ngram_size: int = field(default=3)
    early_stopping: bool = field(default=True)
    repetition_penalty: float = field(default=1.3)
    length_penalty: float = field(default=1.2)

    # Miscellaneous
    remove_unused_columns: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    