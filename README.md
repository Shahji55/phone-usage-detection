# phone-usage-detection
Detect active phone usage on input video files. Models used include the yolo11 coco model and a yolo11 model trained on
a custom dataset.

Inactive/non-used phone is detected according to 2 criteria:
1) Phone remains static for a consecutive frames threshold.
2) Phone is detected in lower half of the frame.

Inactive phone implementation can be extended/implemented using ROIs (Region Of Interest) or other logic based on the scenarios present in the input videos.