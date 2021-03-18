import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import os

model = load_model('model_vgg16.h5')
name =['NORMAL','PNEUMONIA']

for i in name:
    print(i)
    dir1 = os.getcwd()
    dir2 = str(dir1) +'\Datasets'+'\\val'+f'\\{i}'
    dir_arr = os.listdir(dir2)
    print(dir_arr)
    for test_image in dir_arr:
        test_image = str(dir1) +'\Datasets'+'\\val'+f'\\{i}\\' + str(test_image)
        img = image.load_img(test_image, target_size=(224,224))
        x = image.img_to_array(img)
        x= np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        classes = model.predict(img_data)
        print(classes)