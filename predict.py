from keras.models import load_model
from keras.preprocessing import  image
from keras.models import model_from_json
from keras.applications.vgg16 import preprocess_input
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
import keras
import os

model = load_model('model_vgg16.h5')
test_folder = os.getcwd() + '\Datasets'+'\\val'

def folderEncoder(folderName):
    if folderName=='COVID':
        return 0
    elif folderName=="NORMAL":
        return 1
    elif folderName=='PNEUMONIA':
        return 2
    else:
        return None

result=[]
for folder1 in os.listdir(test_folder):
    y_true=folderEncoder(folder1)
    for filename in os.listdir(os.path.join(test_folder,folder1)):
#         print(filename)
        img = image.load_img(os.path.join(os.path.join(test_folder,folder1),filename), target_size=(224, 224))
        x = image.img_to_array(img)
#         x = x / 255
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        y_pred=model.predict(img_data)
#         y_pred=loaded_model.predict(img_data)
        y_pred=np.argmax(y_pred, axis=1)
        result.append([y_true,y_pred[0]])

y_pred=y_true=[] #empty list for placeholder

y_true=[y[0] for y in result] # separating true classes
y_pred=[y[1] for y in result] # separating predicted classes
print(y_true,y_pred)

target_classes = ['COVID','NORMAL','PNEUMONIA'] 

print(classification_report(y_true, y_pred, target_names=target_classes))
print(accuracy_score(y_true, y_pred))


cm = pd.DataFrame(data=confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),index=["Actual Covid","Actual Normal", "Actual Pneumonia"],
columns=["Predicted Covid","Predicted Normal", "Predicted Pneumonia"])
sns_plot = sns.heatmap(cm,annot=True,fmt="d")
sns_plot.figure.savefig("output.png")