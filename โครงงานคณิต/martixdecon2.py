import numpy as np
import cv2
import matplotlib.pyplot as plt

def deblur_image(blurred_img, kernel, regularization_strength=0.5):
    kernel_ft = np.fft.fft2(kernel, s=blurred_img.shape)
    blurred_img_ft = np.fft.fft2(blurred_img)
    
    # Wiener deconvolution
    inverse_ft = np.conj(kernel_ft) / (np.abs(kernel_ft)**2 + regularization_strength)
    deblurred_img_ft = blurred_img_ft * inverse_ft
    
    # Inverse Fourier transform to get the deblurred image
    deblurred_img = np.abs(np.fft.ifft2(deblurred_img_ft))
    
    # Clip values to be in the valid image range
    deblurred_img = np.clip(deblurred_img, 0, 255)
    
    return deblurred_img.astype(np.uint8)

# Load your own image
original_image = cv2.imread('368509586_24171238525853754_8409865852492945768_n.jpg', cv2.IMREAD_GRAYSCALE)

# Example blur kernels
motion_kernel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) / 3
box_kernel = np.ones((3, 3), dtype=np.float32) / 9
gaussian_kernel = cv2.getGaussianKernel(9, 2)
gaussian_kernel = gaussian_kernel @ gaussian_kernel.T
median_kernel = np.ones((3, 3), dtype=np.float32) / 9  # Adjust kernel size as needed

# Perform the convolution to get the blurred images
blurred_motion = cv2.filter2D(original_image, -1, motion_kernel)
blurred_box = cv2.filter2D(original_image, -1, box_kernel)
blurred_gaussian = cv2.filter2D(original_image, -1, gaussian_kernel)
blurred_median = cv2.medianBlur(original_image, 3)

# Deblur the images using Wiener deconvolution
deblurred_motion = deblur_image(blurred_motion, motion_kernel)
deblurred_box = deblur_image(blurred_box, box_kernel)
deblurred_gaussian = deblur_image(blurred_gaussian, gaussian_kernel)
deblurred_median = deblur_image(blurred_median, median_kernel)

# Display the images
plt.figure(figsize=(16, 10))

plt.subplot(2, 4, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(blurred_motion, cmap='gray')
plt.title('Motion Blur')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(blurred_box, cmap='gray')
plt.title('Box Blur')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(blurred_gaussian, cmap='gray')
plt.title('Gaussian Blur')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(blurred_median, cmap='gray')
plt.title('Median Blur')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(deblurred_motion, cmap='gray')
plt.title('Deblurred Motion')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(deblurred_box, cmap='gray')
plt.title('Deblurred Box')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(deblurred_median, cmap='gray')
plt.title('deblurred_median')
plt.axis('off')

plt.tight_layout()
plt.show()
