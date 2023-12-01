import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def regularized_deconvolution(blurred_image, kernel, iterations, alpha):
    # Initialize the estimate with the blurred image
    deblurred_image = np.copy(blurred_image)

    # Normalize the kernel to sum to 1
    kernel = kernel / np.sum(kernel)

    for i in range(iterations):
        # Calculate the relative blur (forward operation)
        relative_blur = convolve2d(deblurred_image, kernel, mode='same')

        # Calculate the correction term
        correction = convolve2d(blurred_image / relative_blur, np.flip(kernel), mode='same')

        # Update the estimate with regularization (Tikhonov regularization)
        deblurred_image *= correction / (np.abs(correction) ** 2 + alpha)

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
blurred_image += np.random.normal(0, np.sqrt(noise_var), blurred_image.shape)

# Set the number of iterations for regularized deconvolution
iterations = 50

# Set the regularization parameter alpha (you can adjust this value based on noise level)
alpha = 1e-3

# Perform regularized deconvolution to deblur the image
deblurred_image = regularized_deconvolution(blurred_image, kernel, iterations, alpha)

# Plot the results
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title('Original Image')
plt.imshow(gray_original_image, cmap='gray')

plt.subplot(132)
plt.title('Blurred and Noisy Image')
plt.imshow(blurred_image, cmap='gray')

plt.subplot(133)
plt.title('Deblurred Image (Regularized Deconvolution)')
plt.imshow(deblurred_image, cmap='gray')

plt.tight_layout()
plt.show()
