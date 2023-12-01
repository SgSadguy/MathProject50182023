import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def standard_deconvolution(blurred_image, kernel):
    # Normalize the kernel to sum to 1
    kernel = kernel / np.sum(kernel)

    # Calculate the size difference between the image and kernel
    diff_y = blurred_image.shape[0] - kernel.shape[0]
    diff_x = blurred_image.shape[1] - kernel.shape[1]

    # Pad the kernel to match the size of the blurred image
    padded_kernel = np.pad(kernel, ((0, diff_y), (0, diff_x)), mode='constant')

    # Perform standard deconvolution using the Fourier transform
    deblurred_image = np.fft.ifft2(np.fft.fft2(blurred_image) / np.fft.fft2(padded_kernel)).real

    return deblurred_image

# Example blur kernel
kernel = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]], dtype=np.float32) / 16

# Generate a sample image
original_image = np.zeros((100, 100))
original_image[40:60, 40:60] = 255  # White square in a black background

# Convert the image to grayscale
gray_original_image = Image.fromarray(original_image.astype(np.uint8)).convert('L')

# Convert the grayscale image to a NumPy array and remove extra dimension
gray_original_image = np.array(gray_original_image)
gray_original_image = gray_original_image.squeeze()

# Perform the convolution to get the blurred image
#blurred_image = convolve2d(gray_original_image, kernel, mode='same')
blurred_image = cv2.medianBlur(gray_original_image, 3)

# Perform standard deconvolution
deblurred_standard = standard_deconvolution(blurred_image, kernel)

# Plot the images
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.title('Original Image')
plt.imshow(gray_original_image, cmap='gray')

plt.subplot(132)
plt.title('Blurred Image')
plt.imshow(blurred_image, cmap='gray')

plt.subplot(133)
plt.title('Standard Deconvolution')
plt.imshow(deblurred_standard, cmap='gray')

plt.tight_layout()
plt.show()
