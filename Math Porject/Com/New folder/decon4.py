import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def kernel_inversion(blurred_image, kernel):
    # Calculate the inverse of the kernel
    kernel_inv = np.linalg.pinv(kernel)
    
    # Deconvolve the blurred image using the inverse kernel
    deblurred_image = convolve2d(blurred_image, kernel_inv, mode='same')
    
    return deblurred_image

# Example blur kernel
kernel = np.array([[0.2, 0.5, 0.2],
                   [0.5, 1.0, 0.5],
                   [0.2, 0.5, 0.2]])

# Generate a sample image
original_image = np.zeros((50, 50))
original_image[20:30, 20:30] = 255  # White square in a black background

# Convert the image to grayscale
gray_original_image = Image.fromarray(original_image.astype(np.uint8)).convert('L')

# Convert the grayscale image to a NumPy array and remove extra dimension
gray_original_image = np.array(gray_original_image)
gray_original_image = gray_original_image.squeeze()

# Convert the kernel to a 2D array
kernel_2d = kernel.reshape((kernel.shape[0], kernel.shape[1]))

# Perform the convolution to get the blurred image
blurred_image = convolve2d(gray_original_image, kernel_2d, mode='same')

# Add some random noise to the blurred image
blurred_image += np.random.normal(0, 10, blurred_image.shape)  # Adjust the noise level as needed

# Perform kernel inversion to deblur the image
deblurred_image = kernel_inversion(blurred_image, kernel_2d)

# Plot the results
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title('Original Image')
plt.imshow(gray_original_image, cmap='gray')

plt.subplot(132)
plt.title('Blurred Image')
plt.imshow(blurred_image, cmap='gray')

plt.subplot(133)
plt.title('Deblurred Image')
plt.imshow(deblurred_image, cmap='gray')

plt.tight_layout()
plt.show()
