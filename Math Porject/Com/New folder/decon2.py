import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def lucy_richardson_deconvolution(blurred_image, kernel, iterations):
    # Initialize the estimate with the blurred image
    deblurred_image = np.copy(blurred_image)

    # Normalize the kernel to sum to 1
    kernel = kernel / np.sum(kernel)

    for i in range(iterations):
        # Calculate the relative blur (forward operation)
        relative_blur = convolve2d(deblurred_image, kernel, mode='same')

        # Calculate the ratio between the observed and relative blur
        ratio = blurred_image / relative_blur

        # Calculate the correction term (backward operation)
        correction = convolve2d(ratio, np.flip(kernel), mode='same')

        # Convert the correction array to the same data type as blurred_image
        correction = correction.astype(blurred_image.dtype)

        # Update the estimate
        deblurred_image *= correction

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

# Add Poisson noise to the blurred image
blurred_image = np.random.poisson(blurred_image)

# Set the number of iterations for Lucy-Richardson deconvolution
iterations = 50

# Perform Lucy-Richardson deconvolution to deblur the image
deblurred_image = lucy_richardson_deconvolution(blurred_image, kernel, iterations)

# Plot the results
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title('Original Image')
plt.imshow(gray_original_image, cmap='gray')

plt.subplot(132)
plt.title('Blurred and Noisy Image')
plt.imshow(blurred_image, cmap='gray')

plt.subplot(133)
plt.title('Deblurred Image (Lucy-Richardson Deconvolution)')
plt.imshow(deblurred_image, cmap='gray')

plt.tight_layout()
plt.show()
