# #Contrast Limited Adaptive Histrogram Equalization

import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread("/home/ptk/kmitl/year3/machine_learning/clahe/images/ct_scan_low_contrast.jpg", 1)

# Convert the image from BGR to LAB color space
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Split the LAB image into its channels
l, a, b = cv2.split(lab_img)

# Apply CLAHE to the L channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

# Merge the CLAHE enhanced L channel back with the A and B channels
l_img = cv2.merge((cl, a, b))

# Convert the LAB image back to BGR color space
final_img = cv2.cvtColor(l_img, cv2.COLOR_LAB2BGR)

# Display the original and CLAHE images side by side
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.title('CLAHE Image')
plt.axis('off')

# Plot histograms for the original and CLAHE images
plt.subplot(2, 2, 3)
plt.hist(l.flat, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of Original Image')

plt.subplot(2, 2, 4)
plt.hist(cl.flat, bins=100, color='green', alpha=0.7)
plt.title('Histogram of CLAHE Image')

plt.tight_layout()

# Save the figure as an image file
output_path = "/home/ptk/kmitl/year3/machine_learning/clahe/images/clahe_comparison.png"
plt.savefig(output_path)
print(f"Comparison image saved at {output_path}")

# Optionally, open the saved image using OpenCV (if you want to display it)
comparison_img = cv2.imread(output_path)
cv2.imshow('Comparison', comparison_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


