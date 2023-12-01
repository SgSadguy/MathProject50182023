import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def wiener_deconvolution(blurred_image, kernel, noise_var):
    # Calculate the power spectral density of the blurred image
    psd_blurred = np.abs(np.fft.fft2(blurred_image)) ** 2

    # Calculate the power spectral density of the kernel
    psd_kernel = np.abs(np.fft.fft2(kernel, s=blurred_image.shape)) ** 2

    # Wiener deconvolution filter
    wiener_filter = np.conj(psd_kernel) / (psd_kernel + noise_var / psd_blurred)

    # Apply the Wiener filter in the frequency domain
    deblurred_image = np.fft.ifft2(np.fft.fft2(blurred_image) * wiener_filter).real

    return deblurred_image

# Example blur kernel
kernel = np.array([[0.2, 0.5, 0.2],
                   [0.5, 1.0, 0.5],
                   [0.2, 0.5, 0.2]])

# Generate a sample image
original_image = np.zeros((100, 100))
original_image[40:60, 40:60] = 255  # White square in a black background

# Convert the image to grayscale
gray_original_image = Image.fromarray(original_image.astype(np.uint8)).convert('L')

# Convert the grayscale image to a NumPy array and remove extra dimension
gray_original_image = np.array(gray_original_image)
gray_original_image = gray_original_image.squeeze()

# Perform the convolution to get the blurred image
blurred_image = convolve2d(gray_original_image, kernel, mode='same')

# Add some random noise to the blurred image
noise_var = 100.0  # Adjust the noise variance as needed
blurred_image_with_noise = blurred_image + np.random.normal(0, np.sqrt(noise_var), blurred_image.shape)

# Perform Wiener deconvolution on the blurred image with noise
deblurred_image_with_noise = wiener_deconvolution(blurred_image_with_noise, kernel, noise_var)

# Perform Wiener deconvolution on the original blurred image (without noise)
deblurred_image_without_noise = wiener_deconvolution(blurred_image, kernel, noise_var)

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.title('Original Image')
plt.imshow(gray_original_image, cmap='gray')

plt.subplot(232)
plt.title('Blurred Image (with Noise)')
plt.imshow(blurred_image_with_noise, cmap='gray')

plt.subplot(233)
plt.title('Blurred Image (without Noise)')
plt.imshow(blurred_image, cmap='gray')

plt.subplot(234)
plt.title('Deblurred Image (Wiener Deconvolution with Noise)')
plt.imshow(deblurred_image_with_noise, cmap='gray')

plt.subplot(235)
plt.title('Deblurred Image (Wiener Deconvolution without Noise)')
plt.imshow(deblurred_image_without_noise, cmap='gray')

plt.tight_layout()
plt.show()
