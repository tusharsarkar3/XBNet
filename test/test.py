import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from XBNet.training_utils import training,predict,predict_proba
from XBNet.models import XBNETClassifier
from XBNet.run import run_XBNET

from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

suppress_qt_warnings()
data = pd.read_csv('test\data (2).csv')
print(data.shape)
x_data = data[data.columns[2:-1]]
print(x_data.shape)
y_data = data[data.columns[1]]
le = LabelEncoder()
y_data = np.array(le.fit_transform(y_data))
print(le.classes_)

X_train,X_test,y_train,y_test = train_test_split(x_data.to_numpy(),y_data,test_size = 0.3,random_state = 0)
# model = torch.load("model.pb")
model = XBNETClassifier(X_train,y_train,3) #Model Intialisation
criterion = torch.nn.BCELoss() #Define criterion
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  #Define optimizer
model, acc, lo, val_ac, val_lo = run_XBNET(X_train,X_test,y_train,y_test,model,criterion,optimizer,32,300) #Train model
print(predict_proba(model,x_data)) # Prediction with probabilities
print(model.feature_importances_) #View feature importances
model.save("trained_model.pb") #Save trained model

print(predict_proba(model,x_data.to_numpy()[0,:]).detach().numpy())

ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, predict_proba(model,X_test).detach().numpy())
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('XBNet: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, predict_proba(model,X_test).detach().numpy())
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='XBNet')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
plt.show()
# show the plot
precision, recall, thresholds = precision_recall_curve(y_test, predict_proba(model,X_test).detach().numpy())
#
# #create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()

print(model.feature_importances_)