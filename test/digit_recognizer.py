import torch
import numpy as np
from sklearn.model_selection import train_test_split
from XBNet.training_utils import training,predict
from XBNet.models import XBNETClassifier
from XBNet.run import run_XBNET
import pandas as pd

data = pd.read_csv('test/train.csv')
y=data[['label']].to_numpy()
x=data.loc[:,'pixel0':].to_numpy()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.20, random_state= True,stratify=y)
y_train=y_train.reshape((-1))
y_test=y_test.reshape((-1))
model = XBNETClassifier(x_train,y_train,num_layers=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
m,acc, lo, val_ac, val_lo = run_XBNET(x_train,x_test,y_train,y_test,model,criterion,optimizer,epochs=1,batch_size=32)
model.save("model_dr.pb")