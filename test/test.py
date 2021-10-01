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
data = pd.read_csv('test\Iris (1).csv')
print(data.shape)
x_data = data[data.columns[:-1]]
print(x_data.shape)
y_data = data[data.columns[-1]]
le = LabelEncoder()
y_data = np.array(le.fit_transform(y_data))
print(le.classes_)

X_train,X_test,y_train,y_test = train_test_split(x_data.to_numpy(),y_data,test_size = 0.3,random_state = 0)
# model = torch.load("model.pb")
model = XBNETClassifier(X_train,y_train,2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

m,acc, lo, val_ac, val_lo = run_XBNET(X_train,X_test,y_train,y_test,model,criterion,optimizer,32,30)
print(predict(m,x_data.to_numpy()[0,:]))
print(model.feature_importances_)