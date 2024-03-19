import requests
from PIL import Image
import math
import matplotlib.pyplot as plt
import torch
from transformers import DetrFeatureExtractor, DetrForSegmentation
import itertools
import io
import seaborn as sns
import numpy as np
from transformers.image_transforms import rgb_to_id, id_to_rgb
import cv2
import os

url = "https://i.ibb.co/PwLWCh4/wall.jpg"
im = Image.open(requests.get(url, stream=True).raw)

# Display the original image
plt.imshow(im)
plt.axis('off')
plt.show()

# Load the DETR feature extractor
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")

# Encode the image
encoding = feature_extractor(im, return_tensors="pt")

# Load the DETR segmentation model
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

# Run the model
outputs = model(**encoding)

# Compute the scores, excluding the "no-object" class (the last one)
scores = outputs.logits.softmax(-1)[..., :-1].max(-1)[0]
keep = scores > 0.85
ncols = 5
fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(keep.sum().item() / ncols), figsize=(18, 10))

for line in axs:
    for a in line:
        a.axis('off')

for i, mask in enumerate(outputs.pred_masks[keep].detach().numpy()):
    ax = axs[i // ncols, i % ncols]
    ax.imshow(mask, cmap="cividis")
    ax.axis('off')

fig.tight_layout()
plt.show()

# use the post_process_panoptic method of DetrFeatureExtractor, which expects as input the target size of the predictions
# (which we set here to the image size)
processed_sizes = torch.as_tensor(encoding['pixel_values'].shape[-2:]).unsqueeze(0)
result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

palette = itertools.cycle(sns.color_palette())

# The segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
# We retrieve the ids corresponding to each mask
panoptic_seg_id = rgb_to_id(panoptic_seg)

# Finally we color each mask individually
panoptic_seg[:, :, :] = 0
for id in range(panoptic_seg_id.max() + 1):
    panoptic_seg[panoptic_seg_id == id] = np.asarray(next(palette)) * 255

# Save the segmented image
segmented_image_path = "segmented_image.png"
plt.figure(figsize=(15, 15))
plt.imshow(panoptic_seg)
plt.axis('off')
plt.savefig(segmented_image_path)
plt.close()

# Clone the Detectron2 repository and install it
# !git clone https://github.com/facebookresearch/detectron2.git
# !pip install ./detectron2
# !pip install --upgrade pyyaml
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog
# from google.colab.patches import cv2_imshow
# from copy import deepcopy

# # We extract the segments info and the panoptic result from DETR's prediction
# segments_info = deepcopy(result["segments_info"])
# # Panoptic predictions are stored in a special format png
# panoptic_seg = Image.open(io.BytesIO(result['png_string']))
# final_w, final_h = panoptic_seg.size
# # We convert the png into a segment id map
# panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
# panoptic_seg = torch.from_numpy(rgb_to_id(panoptic_seg))

# # Detectron2 uses a different numbering of COCO classes, here we convert the class ids accordingly
# meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
# for i in range(len(segments_info)):
#     c = segments_info[i]["category_id"]
#     segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

# # Finally we visualize the prediction
# v = Visualizer(np.array(im.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
# v._default_font_size = 20
# v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)

# # Save the final image
# final_image_path = "final_image.png"
# cv2.imwrite(final_image_path, v.get_image()[:, :, ::-1])

# # Display the final image
# cv2_imshow(v.get_image()[:, :, ::-1])
