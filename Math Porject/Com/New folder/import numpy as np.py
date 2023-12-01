import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def standard_deconvolution(blurred_image, kernel):
    # Calculate the size difference between the image and kernel
    diff_y = blurred_image.shape[0] - kernel.shape[0]
    diff_x = blurred_image.shape[1] - kernel.shape[1]

    # Pad the kernel to match the size of the blurred image
    padded_kernel = np.pad(kernel, ((0, diff_y), (0, diff_x)), mode='constant')

    # Perform standard deconvolution using the Fourier transform
    deblurred_image = np.fft.ifft2(np.fft.fft2(blurred_image) / np.fft.fft2(padded_kernel)).real

    return deblurred_image

# Example blur kernels
kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16

# Generate a sample image
original_image = np.zeros((100, 100))
original_image[20:80, 20:80] = 255  # White square in a black background

# Convert the image to grayscale
gray_original_image = Image.fromarray(original_image.astype(np.uint8)).convert('L')

# Convert the grayscale image to a NumPy array and remove extra dimension
gray_original_image = np.array(gray_original_image)
gray_original_image = gray_original_image.squeeze()

# Perform the convolution to get the blurred images
blurred_gaussian = cv2.filter2D(gray_original_image, -1, gaussian_kernel)
blurred_median = cv2.medianBlur(gray_original_image, 3)

# Perform standard deconvolution on the Gaussian blurred image
deblurred_gaussian = standard_deconvolution(blurred_gaussian, gaussian_kernel)

# Perform standard deconvolution on the median-blurred image
deblurred_median = standard_deconvolution(blurred_median, np.ones((3, 3), dtype=np.float32) / 9)

# Plot the images
plt.figure(figsize=(24, 6))

plt.subplot(161)
plt.title('Original Image')
plt.imshow(gray_original_image, cmap='gray')

plt.subplot(162)
plt.title('Gaussian Blur')
plt.imshow(blurred_gaussian, cmap='gray')

plt.subplot(163)
plt.title('Median Blur')
plt.imshow(blurred_median, cmap='gray')

plt.subplot(164)
plt.title('Deblurred (Gaussian)')
plt.imshow(deblurred_gaussian, cmap='gray')

plt.subplot(165)
plt.title('Deblurred (Median)')
plt.imshow(deblurred_median, cmap='gray')

plt.tight_layout()
plt.show()