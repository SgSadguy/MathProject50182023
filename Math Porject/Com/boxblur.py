import cv2
import numpy as np

def gaussian_blur(image):
    # Define the 3x3 Box blur kernel
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.float32) / 9

    # Apply the Gaussian blur using the convolution operation
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

if __name__ == "__main__":
    # Replace 'input.jpg' with the path to your input image
    input_image_path = "image/image001.jpg"
    # Read the input image
    image = cv2.imread(input_image_path)

    blurred_image = gaussian_blur(image)

    # Display the original and blurred images
    cv2.imshow("Original Image", image)
    cv2.imshow("Box Blurred Image", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.imwrite("box_blurred_image.jpg", blurred_image)
print("Box blurred image saved successfully as blurred_image.jpg.")