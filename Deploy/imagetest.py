import numpy as np
from keras.models import load_model
import cv2
import os

# Load the image
image = cv2.imread('TestImages/ek.jpg')  # Replace 'your_image.jpg' with the path to your image file

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding to create a black and white image
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('TestImages/ek_inverted.jpg',binary_image)
# Show the binary image
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# try:
#     imagepath = 'TestImages/yna.jpg'
#     image = cv2.imread(imagepath)

#     if image is not None:
#         # Image loaded successfully, proceed with further processing
#         print("Image loaded successfully.")
#         cv2.imshow('Image',image)
#         cv2.waitKey(0)  # Wait for any key press to close the window
#         cv2.destroyAllWindows()  # Close all OpenCV windows
#         # Your image processing code here
#     else:
#         # Image could not be loaded
#         print(f"Failed to load the image from {imagepath}")

# except Exception as e:
#     # An exception occurred, print the error message
#     print("An exception occurred:")
#     print(e)

# blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
# sharpened_image = cv2.addWeighted(image, 1.5, blurred_image, -0.5, 0)

# # Resize the image and convert to grayscale
# resized = cv2.resize(sharpened_image, (32, 32))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# # Normalize the pixel values to the [0, 1] range
# resized_normalized = gray.astype('float32') / 255.0

# # Expand dimensions to match the input shape expected by the model
# data = np.expand_dims(resized_normalized, axis=0)