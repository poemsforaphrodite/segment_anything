import requests
from PIL import Image
import math
import matplotlib.pyplot as plt
import torch
from transformers import DetrFeatureExtractor, DetrForSegmentation
import itertools
import io
import seaborn as sns
import numpy
from transformers.image_transforms import rgb_to_id, id_to_rgb

url = "https://i.ibb.co/PwLWCh4/wall.jpg"
im = Image.open(requests.get(url, stream=True).raw)

# Display the original image
plt.imshow(im)
plt.axis('off')
#plt.show()

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
#plt.show()

# use the post_process_panoptic method of DetrFeatureExtractor, which expects as input the target size of the predictions
# (which we set here to the image size)
processed_sizes = torch.as_tensor(encoding['pixel_values'].shape[-2:]).unsqueeze(0)
result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

palette = itertools.cycle(sns.color_palette())

# The segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
# We retrieve the ids corresponding to each mask
panoptic_seg_id = rgb_to_id(panoptic_seg)

# Finally we color each mask individually
panoptic_seg[:, :, :] = 0
for id in range(panoptic_seg_id.max() + 1):
    panoptic_seg[panoptic_seg_id == id] = numpy.asarray(next(palette)) * 255
plt.figure(figsize=(15, 15))
plt.imshow(panoptic_seg)
plt.axis('off')

# Save the last image
plt.savefig("segmented_image.png")
#plt.show()

