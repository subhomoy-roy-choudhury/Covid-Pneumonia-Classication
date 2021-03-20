import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt #For Visualization
import pandas as pd             # For handling data

model = load_model('model_vgg16_2.h5')
name =['COVID','NORMAL','PNEUMONIA']

# results
for i in name:
    print(i)
    dir1 = os.getcwd()
    dir2 = str(dir1) +'\Datasets'+'\\val'+f'\\{i}'
    dir_arr = os.listdir(dir2)
    # print(dir_arr)
    for test_image in dir_arr:
        test_image1 = str(dir1) +'\Datasets'+'\\val'+f'\\{i}\\' + str(test_image)
        img = image.load_img(test_image1, target_size=(224,224))
        x = image.img_to_array(img)
        x= np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        classes = model.predict(img_data)
        y_pred=np.argmax(classes, axis=1)
        prediction = name[y_pred[0]]
        print(prediction,classes,test_image)