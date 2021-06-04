import torch
import numpy as np
from xgboost import XGBClassifier
from collections import OrderedDict
from XBNet.Sequential import Seq

class Model(torch.nn.Module):
    def __init__(self, X_values, y_values, num_layers, num_layers_boosted=1, k=2, epsilon=0.001):
        super(Model, self).__init__()
        self.layers = OrderedDict()
        self.boosted_layers = {}
        self.num_layers = num_layers
        self.num_layers_boosted = num_layers_boosted
        self.X = X_values
        self.y = y_values

        self.take_layers_dim()
        self.base_tree()

        self.epsilon = epsilon
        self.k = k

        self.layers[str(0)].weight = torch.nn.Parameter(torch.from_numpy(self.temp.T))


        self.xg = XGBClassifier()

        self.sequential = Seq(self.layers)
        self.sequential.give(self.xg, self.num_layers_boosted)
        self.sigmoid = torch.nn.Sigmoid()


    def get(self, l):
        self.l = l


    def take_layers_dim(self):
        print("Enter dimensions of linear layers: ")
        for i in range(self.num_layers):
            inp = int(input("Enter input dimensions of layer " + str(i + 1)))
            out = int(input("Enter output dimensions of layer " + str(i + 1)))
            set_bias = bool(input("Set bias as True or False"))
            self.layers[str(i)] = torch.nn.Linear(inp, out, bias=set_bias)
            if i == 0:
                self.input_out_dim = out
        print("Enter your last layer ")
        self.ch = int(input("1. Sigmod \n2. Softmax \n3. None \n"))
        if self.ch == 1:
            self.layers[str(self.num_layers)] = torch.nn.Sigmoid()
        elif self.ch == 2:
            self.layers[str(self.num_layers)] = torch.nn.Softmax()
        else:
            pass

    def base_tree(self):
        self.temp1 = XGBClassifier().fit(self.X, self.y).feature_importances_
        self.temp = self.temp1
        for i in range(1, self.input_out_dim):
            self.temp = np.column_stack((self.temp, self.temp1))

    def forward(self, x, train=True):
        x = self.sequential(x, self.l,train)
        return x
