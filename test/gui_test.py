import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from XBNet.training_utils import training,predict
from XBNet.models import XBNETClassifier
from XBNet.run import run_XBNET

from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

suppress_qt_warnings()
column_to_predict = input("Column to classify: ")
data = pd.read_csv(r'test\Iris (1).csv')
n_df = len(data)
label_encoded = {}
imputations = {}
for i in data.columns:
    imputations[i] = data[i].mode()
    if data[i].isnull().sum()/n_df >= 0.15:
        data.drop(i,axis = 1,inplace=True)
    elif data[i].isnull().sum()/n_df < 0.15 and data[i].isnull().sum()/n_df > 0:
        data[i].fillna(data[i].mode(),inplace=True)
        imputations[i] = data[i].mode()
columns_object = list(data.dtypes[data.dtypes==object].index)
for i in columns_object:
    if i != column_to_predict:
        if data[i].nunique()/n_df < 0.4:
            le = LabelEncoder()
            data[i] = le.fit_transform(data[i])
            label_encoded[i] = le
        else:
            data.drop(i,axis=1,inplace=True)

x_data = data.drop(column_to_predict,axis=1).to_numpy()
columns_finally_used = data.drop(column_to_predict,axis=1).columns
print(x_data[0,:])
print("Number of features are: ",x_data.shape[1])


y_data = data[column_to_predict].to_numpy()
if y_data.dtype == object:
    y_label_encoder = LabelEncoder()
    y_data = y_label_encoder.fit_transform(y_data)

print("Number of classes are: ", np.unique(y_data,return_counts=True))

X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size = 0.3,random_state = 0)
# model = torch.load("model.pb")
model = XBNETClassifier(X_train,y_train,2,input_through_cmd=True,inputs_for_gui=[10,4,4,2])

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

m,acc, lo, val_ac, val_lo = run_XBNET(X_train,X_test,y_train,y_test,model,criterion,optimizer,32,300)

print(predict(m, x_data))
print(model.feature_importances_)



def process_for_predict(df,columns,imputations,encodings):
    data = df[columns]
    n = len(data)
    for i in data.columns:
        if data[i].isnull().sum() >0:
            data[i].fillna(imputations[i], inplace=True)
        if i in encodings.keys():
            data[i] = encodings[i].transform(data[i])
    print(predict(m, data.to_numpy()))

process_for_predict(pd.read_csv(r"test\titanic_test.csv"),columns_finally_used,imputations,label_encoded)