# Multi-Class Non-Maximum Suppression (NMS)

A fast and efficient NumPy implementation of multi-class Non-Maximum Suppression (NMS) for object detection tasks. 

## Installation

No installation required! Just copy the `multi_class_nms.py` file into your project.

## Usage

```python
import numpy as np
from multi_class_nms import multi_class_nms

# Example usage
bboxes = np.array([[10, 10, 50, 50], [15, 15, 55, 55], [100, 100, 150, 150]])  # Shape: (N, 4) and [x1, y1, x2, y2]
scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])  # [N, num_classes]

# Apply NMS
filtered_boxes, filtered_scores, class_indices = multi_class_nms(
    bboxes=bboxes,
    scores=scores,
    nms_thresh=0.4,  # IoU threshold
    min_conf=0.3     # Minimum confidence threshold
)
```

## Parameters

- `bboxes` (np.ndarray): Array of shape (N, 4) containing N bounding boxes in [x1, y1, x2, y2] format
- `scores` (np.ndarray): Array of shape (N, num_classes) with confidence scores per class
- `nms_thresh` (float): IoU threshold for NMS (default: 0.4)
- `min_conf` (float): Minimum confidence threshold to filter detections before NMS (default: 0.3)

## Returns
- `sel_bboxes` (np.ndarray): Filtered bounding boxes after NMS
- `sel_scores` (np.ndarray): Corresponding confidence scores
- `cls_id` (np.ndarray): Class indices of the selected bounding boxes


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Author

- [Amit Pant](https://github.com/amitpant7)

## Acknowledgments

- Based on the Fast R-CNN implementation by Ross Girshick
- Used in NanoDet post-processing pipeline 