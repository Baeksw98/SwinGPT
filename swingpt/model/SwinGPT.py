from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast\
                                        
from transformers.generation.utils import GenerateOutput

from swingpt.utils.mm_utils import *
from swingpt.utils.constants import *
from swingpt.utils.box_utils import *
from swingpt.model.builders.swin_encoder_build import build_vision_tower
from swingpt.model.builders.mm_projector_build import build_vision_projector

class SwinGPTConfig(GPT2Config):
    model_type = "SwinGPT"


class SwinGPTModel(GPT2Model):
    config_class = SwinGPTConfig

    def __init__(self, config: GPT2Config):
        super(SwinGPTModel, self).__init__(config)
        self.vocab_size = config.vocab_size

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # Initialize variables based on model_args
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_weight(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_weight(mm_projector_weights, 'mm_projector'))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                raise ValueError("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False` is required.")
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    None,
                    None,
                    None,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SwinGPTForCausalLM(GPT2LMHeadModel):
    config_class = SwinGPTConfig

    def __init__(self, config: SwinGPTConfig):
        super(GPT2LMHeadModel, self).__init__(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        
        self.transformer = SwinGPTModel(config)
        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )

        self.bbox_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 4)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_images:
            # Get special tokens and add to tokenizer 
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN, DEFAULT_BBOX_TOKEN, DEFAULT_CAPTION_TOKEN, 
                                                DEFAULT_DETECT_TOKEN, DEFAULT_TRIGGER_TOKEN, DEFAULT_EOP_TOKEN], special_tokens=True)
            # Resize the token embeds
            self.resize_token_embeddings(len(tokenizer))

            # Assign token IDs for special tokens
            tokenizer.image_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN])[0]
            tokenizer.bbox_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_BBOX_TOKEN])[0]
            tokenizer.caption_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_CAPTION_TOKEN])[0]
            tokenizer.detect_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_DETECT_TOKEN])[0]
            tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TRIGGER_TOKEN])[0]
            tokenizer.eop_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_EOP_TOKEN])[0]

            if num_new_tokens > 0:
                input_embeds = self.get_input_embeddings().weight.data
                output_embeds = self.get_output_embeddings().weight.data

                input_embeds_avg = input_embeds[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeds_avg = output_embeds[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeds[-num_new_tokens:] = input_embeds_avg
                output_embeds[-num_new_tokens:] = output_embeds_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

                if input_embeds.shape == embed_tokens_weight.shape:
                    input_embeds[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeds[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeds.shape}. Number of new tokens: {num_new_tokens}.")
        
        # Save the tokenizer to the model configurations
        self.tokenizer = tokenizer
        
    def get_image_embeds(self, vision_tower, images):  
        # Ensure that vision tower and images exist to process function
        if vision_tower is None or images is None:
            return None
        
        # Handle different formats of images (list or tensor)
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([img for img in images], dim=0)
            _, concat_image_embeds = vision_tower(concat_images)
            concat_image_embeds = self.get_model().mm_projector(concat_image_embeds)
            split_sizes = [img.shape[0] for img in images]
            image_embeds = torch.split(concat_image_embeds, split_sizes, dim=0)
            image_embeds = [img_f.flatten(0, 1).to(self.device) for img_f in image_embeds]
        else:
            _, image_embeds = vision_tower(images)
            image_embeds = self.get_model().mm_projector(image_embeds)
            image_embeds = image_embeds.to(self.device)

        return image_embeds

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, bbox_inputs,
    ):
        # Instantiate vision tower
        vision_tower = self.get_vision_tower()
        
        # None for input embeds if vision tower & images are non-existent (generation)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype, device=attention_mask.device)), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, None, None, None
        
        # Saving the original values.
        _input_ids, _labels = input_ids, labels
        _position_ids, _attention_mask = position_ids, attention_mask

        # Instantiate variables if they do not exist
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) if position_ids is None else position_ids
        labels = torch.full_like(input_ids, IGNORE_INDEX) if labels is None else labels

        # Remove the padding by using attention masks
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        # Get processed image embeds from SwinTransformer (Vision Encoder)
        image_embeds = self.get_image_embeds(vision_tower, images)

        # Get processed bbox embeds from bbox encoder (Two-layer MLP)
        bbox_embeds = None 
        if bbox_inputs is not None and len(bbox_inputs) > 0:
            bbox_embeds = []
            for bbox_input in bbox_inputs:
                if bbox_input.nelement() > 0:  # Check if the tensor is not empty
                    embed = self.bbox_encoder(bbox_input)
                    # Check if the dimension of the embed is 1 and unsqueeze if necessary
                    if embed.dim() == 1:
                        embed = embed.unsqueeze(0)
                    bbox_embeds.append(embed)
                else:
                    # Handle the case where bbox_input is an empty tensor
                    bbox_embeds.append(torch.empty(0, dtype=bbox_input.dtype, device=bbox_input.device))

        # Initialize places to store input embeds and labels
        final_input_embeds, final_labels, final_input_ids = [], [], []

        for batch_idx, cur_input_ids in enumerate(input_ids):
            # Get current labels
            cur_labels = labels[batch_idx]

            # Prepare indices for splitting input_ids around <image> and <bbox> tokens
            image_indices = (cur_input_ids == self.tokenizer.image_token_id).nonzero(as_tuple=True)[0]
            bbox_indices = (cur_input_ids == self.tokenizer.bbox_token_id).nonzero(as_tuple=True)[0]
            
            # Collect all indices from both image and bbox and sort them
            special_indices = sorted(torch.cat((image_indices, bbox_indices)).tolist())
            
            # Initialize variables of indices and result storage
            cur_index, bbox_idx_counter = 0, 0 
            cur_input_embeds, cur_new_labels = [], []
            cur_new_input_ids = [] 
            
            # Handle chunks of text split by special tokens (image & bbox)
            for special_index in special_indices:
                # Append text embedding 
                if cur_index != special_index:
                    text_segment = cur_input_ids[cur_index:special_index]
                    text_embeds = self.get_model().wte(text_segment)  
                    cur_input_embeds.append(text_embeds)
                    cur_new_labels.append(cur_labels[cur_index:special_index])
                    cur_new_input_ids.append(cur_input_ids[cur_index:special_index])

                # Append image embedding (assume only one image is processed)
                if special_index in image_indices:
                    cur_image_embeds = image_embeds[batch_idx]
                    cur_input_embeds.append(cur_image_embeds)
                    cur_new_labels.append(torch.full((cur_image_embeds.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_input_ids.append(torch.full((cur_image_embeds.shape[0],), IGNORE_INDEX, device=cur_input_ids.device, dtype=cur_input_ids.dtype))
                    
                # Append bbox embedding (use counter to account for multiple bboxes)
                elif special_index in bbox_indices and bbox_embeds is not None:
                    # batched case
                    if len(bbox_inputs) > 1 and len(input_ids) > 1:
                        cur_bbox_embed = bbox_embeds[batch_idx][bbox_idx_counter]
                        cur_input_embeds.append(cur_bbox_embed.unsqueeze(0))
                    # Single case 
                    else: 
                        cur_bbox_embed = bbox_embeds[bbox_idx_counter]
                        cur_input_embeds.append(cur_bbox_embed)
                    
                    # Check if <bbox> comes after <caption>
                    if cur_input_ids[special_index-1] == self.tokenizer.caption_token_id or cur_input_ids[special_index-2] == self.tokenizer.caption_token_id: 
                        # <bbox> following <caption> signals human input, append IGNORE_INDEX to labels
                        cur_new_labels.append(torch.full((1,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    else:
                        # If not, it is gpt output, so append the bbox_token_id to labels
                        cur_new_labels.append(torch.full((1,), self.tokenizer.bbox_token_id, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_input_ids.append(torch.full((1,), self.tokenizer.bbox_token_id, device=cur_labels.device, dtype=cur_labels.dtype))
                    bbox_idx_counter += 1
                cur_index = special_index + 1

            # Append remaining text after the last special token
            if cur_index < len(cur_input_ids):
                text_segment = cur_input_ids[cur_index:]
                text_embeds = self.get_model().wte(text_segment)
                cur_input_embeds.append(text_embeds)
                cur_new_labels.append(cur_labels[cur_index:])
                cur_new_input_ids.append(cur_input_ids[cur_index:])
            
            # Assert that cur_input_embeds and cur_new_labels are of the same torch size
            assert torch.cat(cur_input_embeds, dim=0).size()[0] == torch.cat(cur_new_labels, dim=0).size()[0]

            # Concatenate all embeds and labels for the current input_ids
            final_input_embeds.append(torch.cat(cur_input_embeds, dim=0))
            final_labels.append(torch.cat(cur_new_labels, dim=0))
            final_input_ids.append(torch.cat(cur_new_input_ids, dim=0))
            
        # Truncate sequences to max length as image embeds can make the sequence longer
        if self.tokenizer.model_max_length is not None:
            final_input_embeds = [item[:self.tokenizer.model_max_length] for item in final_input_embeds]
            final_labels = [item[:self.tokenizer.model_max_length] for item in final_labels]
            final_input_ids = [item[:self.tokenizer.model_max_length] for item in final_input_ids]

        # Get max length of input embeds and its batch_size
        max_len = max(item.shape[0] for item in final_input_embeds)
        batch_size = len(final_input_embeds)

        # Get padding of new input embeds and labels
        input_embeds_padded, input_ids_padded = [], []
        labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=final_labels[0].dtype, device=final_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        
        # Configure padding side from tokenizer or default to 'right'
        padding_side = getattr(self.config, 'tokenizer_padding_side', 'right')
        
        # Go through each input_embeds and labels to identify places to add paddings ('pad_token_id' is used)
        for i, (cur_new_embed, cur_new_labels, cur_input_ids) in enumerate(zip(final_input_embeds, final_labels, final_input_ids)):
            cur_len = cur_new_embed.shape[0]
            pad_size = max_len - cur_len
            input_embed_pad_tensor = torch.full((pad_size, cur_new_embed.shape[1]), self.tokenizer.pad_token_id, dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            input_ids_pad_tensor = torch.full((pad_size, 1), self.tokenizer.pad_token_id, dtype=cur_input_ids.dtype, device=cur_input_ids.device)

            if padding_side == "left":
                input_embeds_padded.append(torch.cat((input_embed_pad_tensor, cur_new_embed), dim=0))
                input_ids_padded.append(torch.cat((input_ids_pad_tensor.squeeze(1), cur_input_ids), dim=0))
                if cur_len > 0:
                    labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                input_embeds_padded.append(torch.cat((cur_new_embed, input_embed_pad_tensor), dim=0))
                input_ids_padded.append(torch.cat((cur_input_ids, input_ids_pad_tensor.squeeze(1)), dim=0))
                if cur_len > 0:
                    labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            
        # Use input_embeds_padded and labels_padded as final outcomes 
        final_input_embeds = torch.stack(input_embeds_padded, dim=0)
        final_input_ids = torch.stack(input_ids_padded, dim=0)
        final_labels = labels_padded

        # Finalize the outputs using conditional statement 
        attention_mask = None if _attention_mask is None else attention_mask
        position_ids = None if _position_ids is None else position_ids

        return None, position_ids, attention_mask, past_key_values, final_input_embeds, final_labels, bbox_embeds, final_input_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        bbox_inputs = None, # List
        bbox_labels = None, # List
    ) -> Union[Tuple, CausalLMOutputWithPast]:
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, bbox_embeds, final_input_ids) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, bbox_inputs)

        # Forward pass through the transformer (GPT2 Decoder)
        outputs = self.transformer(
            input_ids=input_ids, 
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            position_ids=position_ids,
        )

        # Take hidden_states from outputs, and get lm_logits
        last_hidden_state = outputs.hidden_states[-1]
        lm_logits = self.lm_head(last_hidden_state)
        
        # Initialize lm_loss and cycle loss
        lm_loss, cycle_loss = None, None
        
        # Calculates lm loss for a text generation task
        if labels is not None:
            lm_loss = self.calculate_lm_loss(lm_logits=lm_logits, labels=labels)

        # Checks if bbox_labels are not empty and have the correct dimensions
        if bbox_labels is not None and len(bbox_labels) > 0 and bbox_embeds is not None and len(bbox_embeds) > 0:
            cycle_loss = self.calculate_cycle_loss(
                last_hidden_state=last_hidden_state, input_ids=final_input_ids, labels=labels, bbox_labels=bbox_labels, bbox_inputs=bbox_inputs, bbox_embeds=bbox_embeds
                )
            
        # Accumulate the losses
        total_loss = 0
        if lm_loss is not None:
            total_loss += lm_loss
        if cycle_loss is not None:
            total_loss += cycle_loss
        total_loss = None if total_loss == 0 else total_loss

        # Structure the return value
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )          

    def calculate_lm_loss(self, lm_logits, labels):
        # Shift the labels to the right
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Calculate the loss using CrossEntropyLoss
        loss_function=CrossEntropyLoss()
        return loss_function(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    def calculate_cycle_loss(self, last_hidden_state, input_ids, labels, bbox_labels, bbox_inputs, bbox_embeds):
        # Initialize loss storage variables
        cycle_loss1, cycle_loss2, box_loss = 0, 0, 0

        # Concatenate the bbox embeds and labels
        bbox_embeds = torch.cat(bbox_embeds, dim=0)
        bbox_labels = torch.cat(bbox_labels, dim=0)
        bbox_inputs = torch.cat(bbox_inputs, dim=0)

        # Get hidden states and necessary inputs
        last_hidden_state = last_hidden_state.view(-1, last_hidden_state.size(-1))
        
        # Use <trigger> token to store bbox information
        bbox_positions = ((input_ids.flatten() == self.tokenizer.trigger_token_id) & (labels.flatten()>0)).nonzero().flatten()
        selected_hidden_states = last_hidden_state[bbox_positions]
        pred_bbox = self.bbox_decoder(selected_hidden_states)
        pred_output_embeds = self.bbox_encoder(pred_bbox)
        pred_input_bbox = self.bbox_decoder(bbox_embeds)
        
        # Compute for box_loss, cycle_loss1 & 2         
        box_loss = self.get_box_loss(pred_bbox=pred_bbox, bbox_labels=bbox_labels)
        cycle_loss1 = F.mse_loss(pred_output_embeds, selected_hidden_states, reduction="none")
        cycle_loss1 = self.get_masked_loss(loss=cycle_loss1, num=1)
        cycle_loss2 = F.l1_loss(pred_input_bbox, bbox_inputs, reduction="none")
        cycle_loss2 = self.get_masked_loss(loss=cycle_loss2, num=1)
    
        # Return the total loss as cycle loss
        return box_loss + cycle_loss1 + cycle_loss2
    
    def get_box_loss(self, pred_bbox, bbox_labels):
        # Check if bbox_labels is empty and return zero loss if true
        if bbox_labels.nelement() == 0:
            return torch.tensor(0.0, device=pred_bbox.device)        
        
        # Calculate the L1 loss for bounding boxes
        loss_bbox = F.l1_loss(pred_bbox, bbox_labels, reduction='none')
        loss_bbox = self.get_masked_loss(loss=loss_bbox, num=1)

        # Check if the bbox coordinates are valid by ensuring x2 >= x1 and y2 >= y1
        valid_mask_pred = (pred_bbox[:, 2:] >= pred_bbox[:, :2]).all(-1)

        masked_pred_bbox = pred_bbox[valid_mask_pred]
        masked_bbox_labels = bbox_labels[valid_mask_pred]

        # Calculate the generalized IoU loss for valid bboxes
        loss_giou = 1 - torch.diag(generalized_box_iou(boxes1=masked_pred_bbox, boxes2=masked_bbox_labels))
        loss_giou = self.get_masked_loss(loss=loss_giou, num=1)
        
        # Return a combination of both losses (L1 and GIoU) with alpha and beta weights
        total_loss = (2 * loss_bbox + 0.2 * loss_giou)
        return total_loss
    
    def get_masked_loss(self, loss, num):
        # Mask the loss for the last num elements
        mask = torch.ones_like(loss)
        mask[-num:] = 1e-10
        return (loss * mask).sum() / mask.sum()
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        images = kwargs.pop("images", None)
        labels = kwargs.pop("labels", None)
        bbox_inputs = kwargs.pop("bbox_inputs", None)
        bbox_labels = kwargs.pop("bbox_labels", None)

        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)
        
        # Filter padding from bbox inputs and labels
        filtered_bbox_inputs = remove_padding_from_bboxes(bbox_inputs) if bbox_inputs is not None else None
        filtered_bbox_labels = remove_padding_from_bboxes(bbox_labels) if bbox_labels is not None else None

        # Append results into inputs dictionary
        inputs['images'] = images
        inputs['labels'] = labels
        inputs['bbox_inputs'] = filtered_bbox_inputs
        inputs['bbox_labels'] = filtered_bbox_labels        
        return inputs

    @torch.no_grad()
    def generate(self, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        # Instantiate necessary components
        input_ids = kwargs.pop("input_ids", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        # Get bbox and transform 
        bbox_inputs = kwargs.pop("bbox_inputs", None)
        bbox_labels = kwargs.pop("bbox_labels", None)
        
        # Check if bbox_inputs and bbox_labels are not None and not empty before padding
        if bbox_inputs is not None and bbox_inputs.numel() > 0:
            padded_bbox_inputs = pad_bboxes(bbox_inputs)
        else:
            padded_bbox_inputs = None        
            
        if bbox_labels is not None and bbox_labels.numel() > 0:
            padded_bbox_labels = pad_bboxes(bbox_labels)
        else:
            padded_bbox_labels = None
            
        # Save the padded bbox_inputs and bbox_labels in kwargs
        kwargs['bbox_inputs'] = padded_bbox_inputs if padded_bbox_inputs is not None else None
        kwargs['bbox_labels'] = padded_bbox_labels if padded_bbox_inputs is not None else None
    
        return super().generate(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
    @torch.no_grad()
    def generate_bbox(self, **kwargs):

        # Get input ids
        input_ids = kwargs.pop("input_ids", None)
        labels = kwargs.pop("labels", None)
        attention_mask = kwargs.pop("attention_mask", None)
        use_cache = kwargs.pop("use_cache", None)
        images = kwargs.pop("images", None)
        bbox_inputs = kwargs.pop("bbox_inputs", None)

        input_ids, position_ids, attention_mask, _, new_inputs_embeds, new_labels, _, new_input_ids = self.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids, 
            labels=labels, 
            attention_mask=attention_mask,
            images=images, 
            bbox_inputs=bbox_inputs, 
            position_ids=None, 
            past_key_values=None, 
        )
        outputs = self.transformer(
            input_ids=input_ids, 
            inputs_embeds= new_inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        last_hidden_state = last_hidden_state.view(-1, last_hidden_state.size(-1))
        pred_bboxes = None
        for new_input, new_label in zip(new_input_ids, new_labels):
            # Initialize bbox position using <trigger> token   
            bbox_positions = ((new_input == self.tokenizer.trigger_token_id) & (new_label>0)).nonzero().view(-1)
            
            # Only process for non-zero bbox positions 
            if bbox_positions.nelement() != 0:  
                selected_hidden_states = last_hidden_state[bbox_positions]
                pred_bboxes = self.bbox_decoder(selected_hidden_states)

        return pred_bboxes
