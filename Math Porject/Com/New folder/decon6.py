import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def deconvolution_inverse(blurred_image, kernel):
    # Calculate the size difference between the image and kernel
    diff_y = blurred_image.shape[0] - kernel.shape[0]
    diff_x = blurred_image.shape[1] - kernel.shape[1]

    # Pad the kernel to match the size of the blurred image
    padded_kernel = np.pad(kernel, ((0, diff_y), (0, diff_x)), mode='constant')

    # Compute the 2D discrete Fourier transform of the padded kernel
    kernel_fft = np.fft.fft2(padded_kernel, s=blurred_image.shape)

    # Compute the 2D discrete Fourier transform of the blurred image
    blurred_image_fft = np.fft.fft2(blurred_image)

    # Divide the Fourier transforms of the blurred image by the kernel
    deconvolved_image_fft = blurred_image_fft / kernel_fft

    # Compute the inverse Fourier transform to get the deblurred image
    deblurred_image = np.fft.ifft2(deconvolved_image_fft).real

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
gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
blurred_gaussian = cv2.filter2D(gray_original_image, -1, gaussian_kernel)
# Add median blur
blurred_median = cv2.medianBlur(gray_original_image, 3)

# Perform deconvolution using inverse of the convolution matrix
deblurred_gaussian = deconvolution_inverse(blurred_gaussian, gaussian_kernel)
deblurred_median = deconvolution_inverse(blurred_median, np.ones((3, 3), dtype=np.float32) / 9)

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
