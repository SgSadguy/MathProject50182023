import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return kernel

def deblur_image(blurred_img, kernel, regularization_strength=0.0001):
    kernel_ft = np.fft.fft2(kernel, s=blurred_img.shape)
    blurred_img_ft = np.fft.fft2(blurred_img)
    
    # Matrix inversion using Wiener deconvolution
    inverse_ft = np.conj(kernel_ft) / (np.abs(kernel_ft)**2 + regularization_strength)
    deblurred_img_ft = blurred_img_ft * inverse_ft
    
    # Inverse Fourier transform to get the deblurred image
    deblurred_img = np.abs(np.fft.ifft2(deblurred_img_ft))
    return deblurred_img

# Load the blurred image
blurred_image = cv2.imread('image/box_blurred_image.jpg', cv2.IMREAD_GRAYSCALE)
#blurred_image = cv2.imread('image/gaussian_blurred_image.tif', cv2.IMREAD_GRAYSCALE)


# Define a Gaussian kernel for blurring
kernel_size = 18
sigma = 2.0
blur_kernel = create_gaussian_kernel(kernel_size, sigma)

# Deblur the image using matrix inversion
deblurred_image = deblur_image(blurred_image, blur_kernel)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(deblurred_image, cmap='gray')
plt.title('Deblurred Image')
plt.axis('off')

plt.show()