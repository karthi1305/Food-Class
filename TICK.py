import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import os
import matplotlib.pyplot as plt

# Load the model from JSON and HDF5 files
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

def classif(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model.predict(test_image)
    predictions = result[0]
    labels = ['burger', 'chicken briyani', 'dosa', 'idly', 'pizza', 'pongal', 'poori', 'white rice']

    # Find the predicted class index
    predicted_class_index = np.argmax(predictions)
    predicted_class = labels[predicted_class_index]
    
    print('Predicted Class:', predicted_class)

    # Display the image using Matplotlib
    img = plt.imread(img_name)
    plt.imshow(img)
    plt.title(predicted_class)
    plt.show()

try:
    import os
    path = 'data/test'
    files = []
    for r, d, f in os.walk(path):
       for file in f:
         if '.jpg' in file:
           files.append(os.path.join(r, file))

    for f in files:
        classif(f)
        print('\n')
except Exception as e:
    print(e)
