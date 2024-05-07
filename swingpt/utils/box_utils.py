import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_area
from swingpt.utils.constants import *
import numpy as np 

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """ Calculate the Intersection over Union (IoU) between two sets of boxes. """
    # Calculate the area of each box
    area1 = box_area(boxes1)  
    area2 = box_area(boxes2)  

    # Find intersections
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  
    
    # Width and height of intersection area, clamped to 0 if negative
    width_height = (right_bottom - left_top).clamp(min=0)  

    # Compute intersection and union
    intersection = width_height[:, :, 0] * width_height[:, :, 1]  
    union = area1[:, None] + area2 - intersection

    # Compute IoU
    iou = intersection / union

    return iou, union

def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Generalized Intersection over Union (G-IoU) between two sets of boxes.
    The bounding box format should be (x_min, y_min, x_max, y_max).
    """
    # Ensure all boxes are properly formed, i.e., bottom-right corner is greater than top-left corner
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "Invalid boxes in boxes1 with x_max < x_min or y_max < y_min"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "Invalid boxes in boxes2 with x_max < x_min or y_max < y_min"

    # Compute the Intersection over Union (IoU) and the union area of the boxes
    iou, union = box_iou(boxes1, boxes2)

    # Compute the smallest enclosing box
    left_top = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # Calculate width and height of the enclosing box and clamp negative values to zero
    width_height = (right_bottom - left_top).clamp(min=0)
    enclosing_area = width_height[:, :, 0] * width_height[:, :, 1]

    # Calculate the Generalized IoU
    generalized_iou = iou - (enclosing_area - union) / enclosing_area

    return generalized_iou

def box_xywh_to_xyxy(box, width=None, height=None):
    """ Convert bbox from (x, y, w, h) to (x1, y1, x2, y2) and clamp to image dimensions """
    assert width is None or width > 0, "Width must be None or a positive integer"
    assert height is None or height > 0, "Height must be None or a positive integer"

    # Unpack the bounding box coordinates
    x, y, bw, bh = box.unbind(-1)
    x2 = x + bw
    y2 = y + bh

    # Clamp to the image dimensions if specified
    if width is not None:
        x2 = torch.min(x2, torch.tensor(width, device=box.device))
    if height is not None:
        y2 = torch.min(y2, torch.tensor(height, device=box.device))

    # Stack all processed bounding boxes
    return torch.stack([x, y, x2, y2], dim=-1)

def normalize_box_xyxy(box, width, height):
    """ Normalize bbox (x1, y1, x2, y2) with image width and height """
    assert width > 0 and height > 0, "Image dimensions must be positive."

    # Unbind the coordinates for easier manipulation
    x1, y1, x2, y2 = box.unbind(-1)

    # Normalize the coordinates
    norm_x1 = torch.clamp(x1 / width, 0.0, 1.0)
    norm_y1 = torch.clamp(y1 / height, 0.0, 1.0)
    norm_x2 = torch.clamp(x2 / width, 0.0, 1.0)
    norm_y2 = torch.clamp(y2 / height, 0.0, 1.0)

    # Stack all processed bounding boxes
    return torch.stack([norm_x1, norm_y1, norm_x2, norm_y2], dim=-1)

def preprocess_bbox(bbox_inputs, width, height, normalize=True):
    """ 
    Processes bounding box coordinates to convert them from xywh format to xyxy format 
    An option to normalize the coordinates based on image dimensions. 
    """
    # Return as is if there are no bounding boxes
    if bbox_inputs.size(0) == 0:
        return bbox_inputs  

    # Process each bounding box
    processed_bboxes = []
    for bbox in bbox_inputs:
        # Convert from xywh to xyxy format
        bbox_xyxy = box_xywh_to_xyxy(bbox, width=width, height=height)

        # Normalize the bounding box coordinates if required
        if normalize:
            bbox_xyxy = normalize_box_xyxy(bbox_xyxy, width=width, height=height)
        processed_bboxes.append(bbox_xyxy)

    # Stack all processed bounding boxes into a single tensor
    return torch.stack(processed_bboxes, dim=0)

def denormalize_box_xyxy(bbox, width, height):
    """Convert normalized bbox [x1, y1, x2, y2] to absolute pixel coordinates."""
    return [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]

def box_xyxy_to_xywh(bbox):
    """Convert bbox from [x1, y1, x2, y2] to [x, y, width, height]."""
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

def pad_bboxes(bboxes):
    """Pads a list of bboxes to the size of the largest bboxes in the list."""
    # Check if tensor is empty or all rows are zeros
    if bboxes.numel() == 0 or (bboxes == 0).all(dim=1).all():
        return torch.empty((0, 4), device='cuda:0')
    else:
        print("Bounding boxes are valid. Proceed")
    
    # Find the maximum number of rows (first dimension)
    max_rows = max(bbox.size(0) for bbox in bboxes)

    padded_bboxes = []
    for bbox in bboxes:
        # Handle for empty bboxes
        if bbox.numel() == 0:
            # Create a new tensor with same bbox shape with padded value 
            new_bbox = torch.full((max_rows, bbox.shape[1] if len(bbox.shape) > 1 else 4), IGNORE_INDEX, dtype=bbox.dtype, device=bbox.device)
            padded_bboxes.append(new_bbox)
        elif bbox.size(0) < max_rows:
            # Calculate the number of rows to pad at the end of the tensor
            pad_size = (0, 0, 0, max_rows - bbox.size(0))
            padded_bboxes.append(F.pad(bbox, pad=pad_size, value=IGNORE_INDEX))
        else:
            padded_bboxes.append(bbox)

    # Stack the results to create a tensor with shape [bbox count, max rows, 4]
    stacked_bboxes = torch.stack(padded_bboxes)

    # Ensure the tensor is always three-dimensional even for [1, rows, 4]
    if stacked_bboxes.dim() == 2:
        # Add the middle dimension if missing
        stacked_bboxes = stacked_bboxes.unsqueeze(1)  

    return stacked_bboxes

def remove_padding_from_bboxes(bboxes):
    """Removes rows filled with the specified padding value from each tensor in a batch of bounding box tensors."""
    # Create a mask that identifies rows where all elements are the padding value
    valid_rows_mask = ~(bboxes == IGNORE_INDEX).all(dim=2)
    
    # Filter rows for each tensor in the batch using the mask
    filtered_bboxes = [bbox[mask] for bbox, mask in zip(bboxes, valid_rows_mask)]

    # Return a list of tensors with the padding removed
    return filtered_bboxes

