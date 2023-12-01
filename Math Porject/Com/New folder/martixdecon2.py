import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def standard_deconvolution(blurred_image, kernel):
    # Normalize the kernel to sum to 1
    #kernel = kernel / np.sum(kernel)

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
motion_kernel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) / 3
box_kernel = np.ones((3, 3), dtype=np.float32) / 9

# Generate a sample image
original_image = np.zeros((100, 100))
original_image[20:80, 20:80] = 255  # White square in a black background

# Convert the image to grayscale
gray_original_image = Image.fromarray(original_image.astype(np.uint8)).convert('L')

# Convert the grayscale image to a NumPy array and remove extra dimension
gray_original_image = np.array(gray_original_image)
gray_original_image = gray_original_image.squeeze()

# Perform the convolution to get the blurred images
blurred_motion = cv2.filter2D(gray_original_image, -1, motion_kernel)
blurred_box = cv2.filter2D(gray_original_image, -1, box_kernel)

# Perform standard deconvolution
deblurred_motion = standard_deconvolution(blurred_motion, motion_kernel)
deblurred_box = standard_deconvolution(blurred_box, box_kernel)

plt.figure(figsize=(18, 4))

plt.subplot(151)
plt.title('Original Image')
plt.imshow(gray_original_image, cmap='gray')
plt.savefig('original_image.png')  # Save the original image

plt.subplot(153)
plt.title('Box Blur')
plt.imshow(blurred_box, cmap='gray')
plt.savefig('blurred_box.png')  # Save the box-blurred image

plt.subplot(152)
plt.title('Motion Blur')
plt.imshow(blurred_motion, cmap='gray')
plt.savefig('blurred_motion.png')  # Save the motion-blurred image

plt.subplot(155)
plt.title('Deblurred (Box)')
plt.imshow(deblurred_box, cmap='gray')
plt.savefig('deblurred_box.png')  # Save the deblurred box image

plt.subplot(154)
plt.title('Deblurred (Motion)')
plt.imshow(deblurred_motion, cmap='gray')
plt.savefig('deblurred_motion.png')  # Save the deblurred motion image

plt.tight_layout()
plt.show()