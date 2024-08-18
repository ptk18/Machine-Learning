import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, color
from skimage.segmentation import mark_boundaries

# Load the image
img = cv2.imread("/home/ptk/kmitl/year3/machine_learning/super-pixel/images/chest_ct.jpg", 1)

# Convert the image from BGR to RGB (for skimage compatibility)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Super-pixel segmentation using SLIC
segments_slic = segmentation.slic(img_rgb, n_segments=250, compactness=10, sigma=1, start_label=1)

# Generate the segmentation map by marking boundaries
segmentation_map = mark_boundaries(img_rgb, segments_slic)

# Display the original image and the segmentation map side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(img_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmentation_map)
ax[1].set_title('Super-pixel Segmentation Map')
ax[1].axis('off')

plt.tight_layout()

# Save the figures as images
original_image_path = "/home/ptk/kmitl/year3/machine_learning/super-pixel/images/original_image.png"
segmentation_map_path = "/home/ptk/kmitl/year3/machine_learning/super-pixel/images/superpixel_segmentation_map.png"

fig.savefig("/home/ptk/kmitl/year3/machine_learning/clahe/superpixel_results.png")
plt.close(fig)  # Close the figure to release memory

# Save each image separately if needed
plt.imsave(original_image_path, img_rgb)
plt.imsave(segmentation_map_path, segmentation_map)

print(f"Original image saved at {original_image_path}")
print(f"Segmentation map saved at {segmentation_map_path}")
print(f"Combined image saved at '/home/ptk/kmitl/year3/machine_learning/clahe/superpixel_results.png'")

# Display the original image and the segmentation map in separate windows
cv2.imshow('Original Image', img)
cv2.imshow('Super-pixel Segmentation Map', segmentation_map)

# Wait for user input to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
