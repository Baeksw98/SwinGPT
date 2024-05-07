import orjson
import re
import torch
from torch.utils.data import DataLoader
from torch.cuda import OutOfMemoryError

from swingpt.datasets.dataset_processor import LazyDataset, DataCollator
from swingpt.utils.train_utils import *
from swingpt.utils.data_utils import * 
from swingpt.utils.eval_utils import * 
from swingpt.utils.inference_utils import * 

from torchmetrics.detection import MeanAveragePrecision

from pytorch_lightning import LightningDataModule, LightningModule

class SwinGPTEvaluator(LightningModule):
    def __init__(self, model, tokenizer, training_args, mode=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.mode = mode 
        self.device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # self.BLEU_TM = BLEUScore(n_gram=4, smooth=False)  
        self.BLEU = BLEUscore() 
        self.CIDEr = CIDERscore()
        self.MAP = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], box_format='xyxy')

        self.model.to(self.device_map)
        if hasattr(self.model, 'get_vision_tower'):
            self.model.get_vision_tower().to(self.device_map)
        
    def forward(self, input_ids, attention_mask=None, labels=None, images=None, bbox_inputs=None, bbox_labels=None):
        # Pass all inputs to the model's forward method. Match these parameters with those expected by your model.
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            images=images, bbox_inputs=bbox_inputs, bbox_labels=bbox_labels, return_dict=True)
        return outputs
    
    def prepare_generation_input(self, batch, index=2):
        input_ids = batch.pop('input_ids', None)
        attention_mask = batch.pop('attention_mask', None)
        labels = batch.pop('labels', None)

        # Sequence to find
        end_of_response_sequence = self.tokenizer.encode('\n###', add_special_tokens=False)
        end_of_response_length = len(end_of_response_sequence)

        new_input_ids, new_attention_masks, new_labels = [], [], []
        new_bbox_inputs, new_images = [], [] 
        # Iterate over each sequence in the batch
        for i in range(input_ids.size(0)):
            # Find end positions of the sequence for this specific input_id
            end_positions = find_sequence_positions(input_ids[i].unsqueeze(0), end_of_response_sequence)
            
            if end_positions is not None and end_positions.numel() >= 3:
                # Take the third occurrence
                last_response_position = end_positions[index].item() + end_of_response_length + 3
                temp_input_ids = input_ids[i, :last_response_position]
                temp_attention_masks = attention_mask[i, :last_response_position]
                temp_labels = labels[i, :last_response_position]
            else:
                # If no third occurrence, append the original input_ids and mask
                temp_input_ids = input_ids[i]
                temp_attention_masks = attention_mask[i]
                temp_labels = labels[i]
            
            # Handle bbox_inputs
            if self.tokenizer.bbox_token_id in temp_input_ids:
                bbox_inputs = batch.get('bbox_inputs', [])
                if isinstance(bbox_inputs, list):
                    bbox_input_item = bbox_inputs[i][0] 
                else:
                    bbox_input_item = bbox_inputs[0] if len(bbox_inputs) > 0 else None
                new_bbox_inputs.append(bbox_input_item.unsqueeze(0))

            # Handle images
            images = batch.get('images', [])
            if self.tokenizer.image_token_id in temp_input_ids:
                if isinstance(images, list):
                    image_item = images[i] 
                else:
                    image_item = images[0] if len(images) > 0 else None
                new_images.append(image_item.unsqueeze(0))

            # Append results per each loop
            new_input_ids.append(temp_input_ids)
            new_attention_masks.append(temp_attention_masks)
            new_labels.append(temp_labels)

        # Convert lists back to tensors
        input_ids = torch.nn.utils.rnn.pad_sequence(new_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(new_attention_masks, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        # Concatenate the bbox_inputs and images if not empty
        new_bbox_inputs = torch.cat(new_bbox_inputs, dim=0) if new_bbox_inputs else torch.empty(0, 4)
        new_images = torch.cat(new_images, dim=0) if new_images else torch.empty(0)
        
        return dict(input_ids=input_ids, attention_mask=attention_mask, 
                    labels=labels, bbox_inputs=new_bbox_inputs, images=new_images)

    def process_image_captions(self, beam_outputs, labels):
        # Regex to match unwanted patterns: '###', ':', 'gpt:', 'human:', and newlines
        unwanted_patterns = re.compile(r'###|:|gpt:|human:|\n')

        preds = []
        targets = []

        for output, label_seq in zip(beam_outputs, labels):
            # Decode entire sequences first
            full_output_text = self.tokenizer.decode(output, skip_special_tokens=True)

            # Find the start of GPT outputs in the decoded strings by matching the 'gpt' token
            gpt_start_pos = full_output_text.find('gpt')

            if gpt_start_pos != -1:
                # Extract from the start position onward and remove unwanted patterns
                pred_text = unwanted_patterns.sub('', full_output_text[gpt_start_pos + len('gpt'):])
                preds.append(pred_text.strip())
            else:
                # Handle cases where 'gpt' is not found
                preds.append(' ')  

            # Filter out IGNORE_INDEX from label sequence before decoding
            valid_label_ids = [id for id in label_seq if id != IGNORE_INDEX]

            # Decode the valid label ids to text, skipping special tokens and remove unwanted patterns
            if valid_label_ids:
                target_text = unwanted_patterns.sub('', self.tokenizer.decode(valid_label_ids, skip_special_tokens=True))
                targets.append(target_text.strip())
            else:
                targets.append('')

        return preds, targets
    
    def process_bounding_boxes(self, outputs, labels, input_ids, bbox_labels, bbox_inputs, bbox_embeds, category_ids):
        # Initialize pred and label bboxes for storage
        pred_bboxes, target_bboxes = [], []

        # Get last hidden states
        last_hidden_state = outputs.hidden_states[-1]
        last_hidden_state = last_hidden_state.view(-1, last_hidden_state.size(-1))
        
        for idx, (cur_input_id, cur_label) in enumerate(zip(input_ids, labels)):
            # Initialize bbox position, current category and bbox label 
            bbox_positions = ((cur_input_id == self.tokenizer.trigger_token_id) & (cur_label > 0)).nonzero().view(-1)
            cur_category = category_ids[idx]
            cur_bbox_label = bbox_labels[idx]
            cur_bbox_input = bbox_inputs[idx]
            cur_bbox_embed = bbox_embeds[idx]
            
            # Only process for non-zero bbox positions 
            if bbox_positions.nelement() != 0:  
                selected_hidden_states = last_hidden_state[bbox_positions]
                cur_bbox_pred = self.model.bbox_decoder(selected_hidden_states)
                pred_output_embeds = self.model.bbox_encoder(cur_bbox_pred)
                pred_input_bbox = self.model.bbox_decoder(cur_bbox_embed)
                
                # Calculating for losses (box_loss + cycle_loss1 + cycle_loss2)
                box_loss = self.model.get_box_loss(pred_bbox=cur_bbox_pred, bbox_labels=cur_bbox_label)
                cycle_loss1 = F.mse_loss(pred_output_embeds, selected_hidden_states, reduction="none")
                cycle_loss1 = self.model.get_masked_loss(loss=cycle_loss1, num=1)
                cycle_loss2 = F.l1_loss(pred_input_bbox, cur_bbox_input, reduction="none")
                cycle_loss2 = self.model.get_masked_loss(loss=cycle_loss2, num=1)
                total_loss = box_loss + cycle_loss1 + cycle_loss2
                
                # Check if the tensor is scalar (zero-dimensional)
                if total_loss.ndimension() == 0:
                    # Unsqueeze to make it a 1D tensor of size 1
                    total_loss = total_loss.unsqueeze(dim=0)  
                
                # Append results to the storage
                pred_bboxes.append({
                    'boxes': cur_bbox_pred,
                    'scores': total_loss,  
                    'labels': cur_category.unsqueeze(dim=0)
                })
                target_bboxes.append({
                    'boxes': cur_bbox_label,
                    'labels': cur_category.unsqueeze(dim=0)
                })
        return pred_bboxes, target_bboxes

    def log_metrics(self, metric_name, metric):
        try:
            if isinstance(metric, torch.Tensor):
                # Ensure it is in floating point format and calculate mean if necessary
                metric = metric.float()
                if metric.numel() > 1:
                    metric = metric.mean()
                    
            self.log(metric_name, metric, on_step=False,
                    on_epoch=True, prog_bar=True, logger=True,
                    batch_size=self.training_args.per_device_eval_batch_size)
        
        except Exception as e:
            print(f"Failed to log {metric_name}: {str(e)}")

    def execute_model(self, batch):
        input_ids = batch.get('input_ids', None)
        labels = batch.get('labels', None)
        attention_mask = batch.get('attention_mask', None)
        bbox_inputs = batch.get('bbox_inputs', None)
        bbox_labels = batch.get('bbox_labels', None)
        images = batch.get('images', None)
        category_ids = batch.pop('category_ids', None)

        # Handle for Image Captioning
        if self.mode == 'IC':
            # Define stopping criteria based on Punctuation marks 
            punctuation_marks = ['.', '?', '!']  
            stopping_criteria = PunctuationStoppingCriteria(punctuation_marks, self.tokenizer)
            
            model_inputs = self.prepare_generation_input(batch, index=1)
            beam_outputs = self.model.generate(
                **model_inputs,
                do_sample=True if self.training_args.temperature > 0 else False,
                temperature=self.training_args.temperature,
                max_new_tokens=self.training_args.max_new_tokens,
                num_beams=self.training_args.num_beams,
                no_repeat_ngram_size=self.training_args.no_repeat_ngram_size,
                early_stopping=self.training_args.early_stopping,
                use_cache=True,
                repetition_penalty=self.training_args.repetition_penalty,
                length_penalty=self.training_args.length_penalty,
                stopping_criteria=[stopping_criteria],
            )
            preds, targets = self.process_image_captions(beam_outputs, labels)
            return preds, targets

        # Handle for Object Detection
        elif self.mode == 'OD':
            _, _, _, _, _, new_labels, new_bbox_embeds, new_input_ids = self.model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids, 
                labels=labels, 
                attention_mask=attention_mask,
                images=images, 
                bbox_inputs=bbox_inputs,
                position_ids=None, 
                past_key_values=None, 
            )
            outputs = self.forward(
                input_ids=input_ids, 
                labels=labels, 
                attention_mask=attention_mask,
                bbox_inputs=bbox_inputs, 
                images=images, 
            )
            # Process bounding boxes detection
            pred_bboxes, target_bboxes = self.process_bounding_boxes(            
                outputs = outputs, 
                labels = new_labels, 
                input_ids = new_input_ids, 
                bbox_labels = bbox_labels, 
                bbox_inputs = bbox_inputs, 
                bbox_embeds = new_bbox_embeds,
                category_ids = category_ids
            )
            return pred_bboxes, target_bboxes

        else:
            raise ValueError("Invalid mode specified. Use 'OD' for Object Detection or 'IC' for Image Captioning.")
        
    def test_step(self, batch, batch_idx):
        """Perform a test step. This method is used by the trainer.test() call and is expected to return the metric outputs."""
        try:
            # Moves all inputs to the same device
            batch = {k: move_to_device(v, self.device_map) for k, v in batch.items()}
            
            # Handling for Image Captioning
            if self.mode == 'IC':
                # Get pred captions and targets
                preds, targets = self.execute_model(batch)

                # Update BLEU@4 and CIDEr
                self.BLEU.update(preds, targets)
                self.CIDEr.update(preds, targets)
                self.log_metrics('BLEU', self.BLEU)
                self.log_metrics('CIDEr', self.CIDEr)

            # Handling for Object Detection
            elif self.mode == 'OD':
                pred_bboxes, target_bboxes = self.execute_model(batch)

                # Update MAP (Average Precision @IOU=0.5)
                self.MAP.update(pred_bboxes, target_bboxes)
                
                # Compute the MAP and extract only the map_50 value
                computed_metrics = self.MAP.compute()
                map_50 = computed_metrics['map_50']
                
                # Log only the map_50 value
                self.log_metrics('MAP_50', map_50)
            else:
                raise ValueError("Invalid mode specified. Use 'OD' for Object Detection or 'IC' for Image Captioning.")

        except OutOfMemoryError:
            print("CUDA Out of Memory error at test step:", batch_idx, "Attempting to clear cache.")
            torch.cuda.empty_cache()  
            print(f"Ran out of memory during validation on step {batch_idx}, cache cleared. Retrying step...")
            
        # Optionally, clear cache every 500 steps
        if batch_idx % 500 == 0:
            torch.cuda.empty_cache()
            print("Cleared CUDA cache at step:", batch_idx)
            
    def on_test_epoch_end(self):
        # Initialize storage
        results = {}
        
        if self.mode == 'IC':
            BLEU_score = self.BLEU.compute()
            CIDEr_score = self.CIDEr.compute()
            results['final_BLEU'] = BLEU_score.item() if isinstance(BLEU_score, torch.Tensor) else BLEU_score
            results['final_CIDER'] = CIDEr_score.item() if isinstance(CIDEr_score, torch.Tensor) else CIDEr_score
            print(f"Final BLEU Score: {BLEU_score}")
            print(f"Final CIDER Score: {CIDEr_score}")
        
        elif self.mode == 'OD':
            computed_metrics = self.MAP.compute()
            map_50 = computed_metrics['map_50']
            
            # Convert map_50 tensor to a scalar if it's a single-element tensor
            map_50_value = map_50.item() if map_50.numel() == 1 else map_50.float().mean().item()

            # Store the map_50 value in a results dictionary
            results = {'final_MAP_50': map_50_value}
            print(f"Final MAP 50: {map_50_value}")
        
        else:
            raise ValueError("Invalid mode specified. Use 'OD' for Object Detection or 'IC' for Image Captioning.")
        
        # Save the final results in JSON file format
        self.save_evaluation_results(results)

    
    def save_evaluation_results(self, results):
        # Define the directory path where the file will be saved
        directory_path = self.training_args.output_dir
        
        # Check if the directory exists, and if not, create it
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
        # Define the file path
        file_path = os.path.join(directory_path, f'{self.mode}_evaluation_results.json')
        
        # Save the final results in JSON file format
        with open(file_path, 'wb') as f:  
            f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))

        # Optionally print a confirmation message
        print(f'Results saved to {file_path}')
                        
    def predict_step(self, batch, batch_idx):
        """Perform a prediction step. This method is expected to return the predictions."""
        # Ensure batch data is on the same device as the model
        batch = {k: move_to_device(v, self.device_map) for k, v in batch.items()}

        # Handling for Image Captioning
        if self.mode == 'IC':
            preds, targets = self.execute_model(batch)
            return preds
        
        # Handling for Object Detection
        elif self.mode == 'OD':            
            # Generate bboxes 
            pred_bboxes = self.model.generate_bbox(**batch)
            return pred_bboxes         
        else:
            raise ValueError("Invalid mode specified. Use 'OD' for Object Detection or 'IC' for Image Captioning.")

class EvaluatorDataModule(LightningDataModule):
    def __init__(self, tokenizer, data_args, training_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.batch_size_val = training_args.per_device_eval_batch_size
        self.data_loader_workers = training_args.dataloader_num_workers
    
    def setup(self, stage=None):
        self.IC_test_dataset = LazyDataset(
            json_file_path=self.data_args.test_IC_path,
            tokenizer=self.tokenizer,
            data_args=self.data_args
        )
        self.OD_test_dataset = LazyDataset(
            json_file_path=self.data_args.test_OD_path,
            tokenizer=self.tokenizer,
            data_args=self.data_args
        )
        self.data_collator = DataCollator(tokenizer=self.tokenizer, is_eval_mode=self.data_args.is_eval_mode)

    def test_dataloader(self, test_type):
        if test_type == 'IC':
            return DataLoader(self.IC_test_dataset, batch_size=self.batch_size_val, shuffle=False,
                            collate_fn=self.data_collator, num_workers=self.data_loader_workers, pin_memory=True)
        elif test_type == 'OD':
            return DataLoader(self.OD_test_dataset, batch_size=self.batch_size_val, shuffle=False,
                            collate_fn=self.data_collator, num_workers=self.data_loader_workers, pin_memory=True)
