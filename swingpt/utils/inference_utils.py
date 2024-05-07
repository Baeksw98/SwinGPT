from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import torch
import copy
import re
import numpy as np

from swingpt.utils.constants import *
from swingpt.utils.box_utils import *
from swingpt.utils.data_utils import *
from swingpt.utils.mm_utils import *
from swingpt.utils import conversation as conversation_lib

def preprocess_sample(sample, model, tokenizer):
    """ Process and tokenize the sample data including images and textual conversations. """
    # Get image and process it 
    image = Image.open(sample['image']).convert("RGB")
    image_width, image_height = image.size
    
    processed_image = process_images([expand2square(image, (0, 0, 0))], model.get_vision_tower().image_processor, model.config)

    # Add signals for the speakers: '### ' at the beginning each sentence, with end signal '\n'
    header = f"{conversation_lib.default_conversation.system}\n"
    conversation = add_speaker_and_signal(header, sample['conversations'])

    # Tokenize the conversations to create input_ids, bbox_inputs, bbox_labels
    result = tokenizer_image_bbox_token(prompt=conversation, tokenizer=tokenizer, return_tensors='pt')
    input_ids, bbox_inputs, bbox_labels = result['input_ids'], result['bbox_inputs'], result['bbox_labels']

    # Preprocess bboxes to get into normalized format
    bbox_inputs = preprocess_bbox(bbox_inputs, image_width, image_height, normalize=True)
    bbox_labels = preprocess_bbox(bbox_labels, image_width, image_height, normalize=True)
    
    # Deep copy for unmasked_labels 
    unmasked_labels = copy.deepcopy(input_ids)

    # Get masked labels 
    input_ids, labels = mask_non_gpt_content(tokenizer, unmasked_labels)
    
    # Collect indices of <trigger> tokens from input_ids
    trigger_indices_input_ids = (input_ids == tokenizer.trigger_token_id).nonzero(as_tuple=True)[0].tolist()

    # Collect indices of <trigger> tokens from masked_labels, assuming it's processed similarly
    trigger_indices_masked_labels = (labels == tokenizer.trigger_token_id).nonzero(as_tuple=True)[0].tolist()

    # Assertion to ensure the trigger indices are the same in both tensors
    assert trigger_indices_input_ids == trigger_indices_masked_labels, "Trigger indices in input_ids and labels do not match."

    # Get attention mask     
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    
    # Prepare the batch
    batch = {
        'images': processed_image,
        'input_ids': input_ids.unsqueeze(dim=0),
        'labels': labels.unsqueeze(dim=0),
        'attention_mask': attention_mask.unsqueeze(dim=0),
        'bbox_inputs': bbox_inputs,
        'bbox_labels': bbox_labels,  
        'trigger_indices': trigger_indices_input_ids,
        'category_ids': sample['category_ids'],      
    }
    return batch

def display_image_with_caption(image, preds):
    """Displays the image with the generated caption."""
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title("Generated Caption: " + ' '.join(preds))
    plt.axis('off')
    plt.show()
    
def truncate_and_mask_input_ids(input_ids, labels, trigger_indices):
    """ Truncate input_ids up to the first index in trigger_indices and create an attention mask. """
    if not trigger_indices:
        raise ValueError("trigger_indices list is empty, cannot truncate input_ids.")
    
    # Truncate input_ids up to the first index from trigger_indices
    truncation_index = trigger_indices[0]
    truncated_input_ids = input_ids[:, :truncation_index + 1]
    truncated_labels = labels[:, :truncation_index + 1]

    # Create an attention mask where all elements are True
    attention_mask = torch.ones_like(truncated_input_ids, dtype=torch.bool)

    return truncated_input_ids, truncated_labels, attention_mask

def extract_tokens(batch, tokenizer):
    """ 
    Extracts specific tokens from a batch of input IDs based on start and end tokens. 
    Returns a list of all texts found between '<detect>' and '<end of prefix>' tokens.
    """    
    # Extract the input_ids from the batch
    input_ids = batch['input_ids'][0].tolist()
    
    # Start and end token IDs from the tokenizer
    detect_token_id = tokenizer.detect_token_id
    eop_token_id = tokenizer.eop_token_id
    
    # Dictionary to store results with sequential keys
    results = {}
    result_index = 1  # Start key index at 1
    
    try:
        start_index = 0
        while start_index < len(input_ids):
            # Search for the next occurrence of '<detect>'
            if detect_token_id in input_ids[start_index:]:
                detect_index = input_ids.index(detect_token_id, start_index) + 1
            else:
                break  # Exit loop if no more '<detect>' found

            # Search for the next occurrence of '<end of prefix>' after '<detect>'
            if eop_token_id in input_ids[detect_index:]:
                eop_index = input_ids.index(eop_token_id, detect_index)
                # Extracting tokens between the detected indices
                tokens = input_ids[detect_index:eop_index]
                decoded_string = tokenizer.decode(tokens)
                
                # Store the result with an incrementing key
                results[result_index] = decoded_string
                
                # Update the key index and start_index for the next search
                result_index += 1
                start_index = eop_index + 1
            else:
                break  # Exit loop if no '<end of prefix>' found after '<detect>'

    except ValueError as e:
        # Handle cases where search may fail unexpectedly
        print("Error extracting tokens:", e)

    return results

def move_to_device(item, device):
    """ Recursively move data to the specified device. """
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [move_to_device(subitem, device) for subitem in item]
    elif isinstance(item, dict):
        return {key: move_to_device(val, device) for key, val in item.items()}
    else:
        return item
    
def get_contrast_color(image_np, bbox):
    """ Obtain the contrast color for the bbox """ 
    center_x, center_y = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)
    avg_color = np.mean(image_np[center_y:center_y + 10, center_x:center_x + 10], axis=(0, 1))
    hsv_color = rgb_to_hsv(avg_color / 255.0)  # Ensure avg_color is scaled to [0, 1]
    hsv_color[2] = 0.5 if hsv_color[2] > 0.5 else 1.0  # Adjust value based on brightness
    contrast_color = hsv_to_rgb(hsv_color)  # This will be in [0, 1] range
    return contrast_color

def annotated_images(image_path, idx, pred_captions=None, pred_bboxes=None, extracted_tokens=None):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)  # Display the image
    ax.axis('off')  # Hide axes
    
    # Set the title based on available data
    title = "(Image Caption) {}".format(pred_captions) if pred_captions is not None else "(Object Detection)"
    plt.title(title, fontsize=14, color='black', backgroundcolor='white')
    
    # Check if there are any bounding boxes to process
    if pred_bboxes is not None and extracted_tokens is not None:
        for index, bbox in enumerate(pred_bboxes):
            # Ensure bbox is a numpy array
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.detach().cpu().numpy()

            # Denormalize and convert bounding box format
            denorm_bbox = denormalize_box_xyxy(bbox, image_width, image_height)
            xywh_bbox = box_xyxy_to_xywh(denorm_bbox)
            
            # Create and add the bounding box patch with red color for visibility
            rect = patches.Rectangle((xywh_bbox[0], xywh_bbox[1]), xywh_bbox[2], xywh_bbox[3],
                                    linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            # Extract and position token text in the center of the bbox
            token_text = extracted_tokens.get(index + 1, '')  # Fetch the token using the 1-based index
            text_x = xywh_bbox[0] + xywh_bbox[2] / 2  # Center horizontally
            text_y = xywh_bbox[1] + xywh_bbox[3] / 2  # Center vertically

            # Annotate the image with extracted tokens if available
            ax.text(text_x, text_y, token_text, fontsize=13, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8))
            
    # Determine the file path
    examples_directory = 'swingpt/demo/examples'
    if not os.path.exists(examples_directory):
        os.makedirs(examples_directory)

    filename = f"{'IC_output_' if pred_captions is not None else 'OD_output_'}{idx}.png"
    filepath = os.path.join(examples_directory, filename)

    # Save the plot as a PNG file with 600 dpi
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    # Show the plot
    plt.show()
    # Close the plot to free up memory
    plt.close(fig)
    