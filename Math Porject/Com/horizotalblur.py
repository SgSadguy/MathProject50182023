import cv2
import numpy as np

def motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel /= kernel_size

    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

if __name__ == "__main__":
    # Replace 'input.jpg' with the path to your input image
    input_image_path = "image/image001.jpg"
    # Read the input image
    image = cv2.imread(input_image_path)

    # Adjust the kernel_size to control the amount of blur
    kernel_size = 15

    blurred_image = motion_blur(image, kernel_size)

    # Save the blurred image as 'output.jpg'
    cv2.imwrite("motion_blurred_image.jpg", blurred_image)
    print("Motion blurred image saved successfully as blurred_image.jpg.")