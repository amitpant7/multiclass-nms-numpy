import numpy as np

# --------------------------------------------------------
# Numpy Implementation of mulit class nms
# Written by Amit Pant
# Used in NanoDet post processing.
# --------------------------------------------------------

import numpy as np
def multi_class_nms(bboxes, scores, nms_thresh=0.4, min_conf=0.3):
    """
    Applies Non-Maximum Suppression (NMS) separately for each class in a multi-class object detection setting.

    Args:
        bboxes (np.ndarray): Array of shape (N, 4) containing N bounding boxes in [x1, y1, x2, y2] format.
        scores (np.ndarray): Array of shape (N, num_classes) with confidence scores per class.
        nms_thresh (float): IoU threshold for NMS.
        min_conf (float): Minimum confidence threshold to filter detections before NMS.

    Returns:
        keep (np.ndarray): Indicies that correspond to filtered bounding boxes in the input array.
        classes (np.ndarray): Class indices of the selected bounding boxes.
        scores (np.ndarray): Corresponding confidence scores.
    """
    num_classes = scores.shape[1]
    
    # Expand bboxes for each class
    bboxes = np.expand_dims(bboxes, axis=1)  # (N, 1, 4)
    bboxes = np.broadcast_to(bboxes, (bboxes.shape[0], num_classes, bboxes.shape[-1]))  # (N, C, 4)

    # Offset bounding boxes by class index to avoid suppressing boxes across classes
    max_val = int(np.max(bboxes) + 100)
    class_offsets = np.arange(num_classes).reshape(1, -1, 1) * max_val
    projected_bboxes = bboxes + class_offsets

    # Filter boxes by confidence
    conf_mask = scores > min_conf
    filtered_bboxes = projected_bboxes[conf_mask]
    filtered_scores = scores[conf_mask]
    
    indecies = np.array(list(zip(*np.nonzero(conf_mask))))
    
    # Apply NMS
    keep_indices = indecies[nms(filtered_bboxes, filtered_scores, nms_thresh)]

    #selecting bbox indicies and class
    keep = keep_indices[:, 0]
    classes = keep_indices[:, 1]
    scores = scores[keep, classes]

    return keep, classes, scores



# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
