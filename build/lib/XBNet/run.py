import torch
from torch.utils.data import Dataset,DataLoader
from XBNet.training_utils import training

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def run_XBNET(X_train,X_test,y_train,y_test,model,criterion,optimizer,batch_size = 16,epochs=100):
    trainDataload = DataLoader(Data(X_train, y_train), batch_size=batch_size)
    testDataload = DataLoader(Data(X_test, y_test), batch_size=batch_size)
    acc, lo, val_ac, val_lo = training(model, trainDataload, testDataload, criterion, optimizer, epochs)
    return model,acc, lo, val_ac, val_lo