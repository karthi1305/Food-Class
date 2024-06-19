
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
import cv2

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

##label=['burger','chicken briyani','dosa','idly','pizza','pongal','poori','white rice']

def classif(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size = (128, 128))
    print('123')
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model.predict(test_image)

    a = np.round(result[0][0])
    b = np.round(result[0][1])
    c = np.round(result[0][2])
    d = np.round(result[0][3])
    e = np.round(result[0][4])
    f = np.round(result[0][5])
    g = np.round(result[0][4])
    h = np.round(result[0][5])

    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
    print(g)
    print(h)



    if a == 1:
        prediction = 'burger'
        print(prediction)
        img=cv2.imread(img_name)
        cv2.imshow(prediction,img)
        print(prediction,img_name)

    elif b == 1:
        prediction = 'chicken briyani'
        print(prediction)
        img=cv2.imread(img_name)
        cv2.imshow(prediction,img)
        print(prediction,img_name)

    elif c == 1:
        prediction = 'dosa'
        print(prediction,img_name)
        print(prediction)
        img=cv2.imread(img_name)
        cv2.imshow('dosa',cv2.resize(img,(400,400)))

    elif d  == 1:
        prediction = 'idly'
        print(prediction,img_name)
        print(prediction)
        img=cv2.imread(img_name)
        cv2.imshow('idly',cv2.resize(img,(400,400)))

    elif e == 1:
        prediction = 'pizza'
        print(prediction,img_name)
        print(prediction)
        img=cv2.imread(img_name)
        cv2.imshow('pizza',cv2.resize(img,(400,400)))

    elif f == 1:
        prediction = 'pongal'
        print(prediction,img_name)
        print(prediction)
        img=cv2.imread(img_name)
        cv2.imshow('pongal',cv2.resize(img,(400,400)))

    elif g == 1:
        prediction = 'poori'
        print(prediction,img_name)
        print(prediction)
        img=cv2.imread(img_name)
        cv2.imshow('poori',cv2.resize(img,(400,400)))

    elif h  == 1:
        prediction = 'white rice'
        print(prediction,img_name)
        print(prediction)
        img=cv2.imread(img_name)
        cv2.imshow('white rice',cv2.resize(img,(400,400)))



try:
    import os
    path = 'data/test'
    files = []
    ##r=root, d=directories, f = files
    for r, d, f in os.walk(path):
       for file in f:
         if '.jpeg' in file:
           files.append(os.path.join(r, file))

    for f in files:
        classif(f)
        print('\n')
except Exception as e:
    print(e)







