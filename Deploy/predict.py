import numpy as np
from keras.models import load_model
import cv2

# List of folders corresponding to classes
Folders = ['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श','ष','स','ह','क्ष','त्र','ज्ञ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

try:
    imagepath = 'TestImages2/Saaa.png'
    image = cv2.imread(imagepath)

    if image is not None:
        resized = cv2.resize(image, (32, 32))
        cv2.imshow('Image',resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Blur and sharpen the image
        blurred_image = cv2.GaussianBlur(resized, (5, 5), 0)
        sharpened_image = cv2.addWeighted(resized, 1.5, blurred_image, -0.5, 0)

        # Resize the image and convert to grayscale
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Normalize the pixel values to the [0, 1] range
        resized_normalized = gray.astype('float32') / 255.0

        # Expand dimensions to match the input shape expected by the model
        data = np.expand_dims(resized_normalized, axis=0)

        # Load the model
        model = load_model('Models/LSTM2adam.h5', compile=False)    

        # Perform prediction
        prediction = model.predict(data)

        # Get the index of the predicted class
        index = np.argmax(prediction)
        predit = Folders[index]
        print(predit)
    else:
        print('Failed to load the image.')

except Exception as e:
    print('Exception occurred:', e)
