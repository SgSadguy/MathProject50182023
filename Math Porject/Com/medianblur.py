import cv2
import numpy as np

def median_blur(image):
    # Apply the median blur using the cv2.medianBlur function
    blurred_image = cv2.medianBlur(image, 3)
    return blurred_image

if __name__ == "__main__":
    # Replace 'input.jpg' with the path to your input image
    input_image_path = "image/image004.jpg"
    # Read the input image
    image = cv2.imread(input_image_path)

    blurred_image = median_blur(image)

    # Display the original and blurred images
    cv2.imshow("Original Image", image)
    cv2.imshow("Median Blurred Image", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.imwrite("median_blurred_image.jpg", blurred_image)
print("Median blurred image saved successfully as blurred_image.jpg.")