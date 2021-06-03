import torch
import numpy as np
import torch.nn.functional as F
from xgboost import XGBClassifier

class Model(torch.nn.Module):
    def __init__(self, X_values, y_values, num_layers, num_layers_boosted=1, epsilon=0.001):
        super(Model, self).__init__()
        self.layers = {}
        self.boosted_layers = {}
        self.num_layers = num_layers
        self.num_layers_boosted = num_layers_boosted
        self.X = X_values
        self.y = y_values

        self.take_layers_dim()
        self.base_tree()

        self.epsilon = epsilon
        self.layers[str(0)].weight = torch.nn.Parameter(torch.from_numpy(self.temp.T))
        self.xg = XGBClassifier()


    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            yield param.weight

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        gen = self._named_members(
            lambda module: self.layers.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

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
        self.ch = int(input("1. Sigmod \n2. Softmax \n3.None \n"))

    def base_tree(self):
        self.temp1 = XGBClassifier().fit(self.X, self.y).feature_importances_
        self.temp = self.temp1
        for i in range(1, self.input_out_dim):
            self.temp = np.column_stack((self.temp, self.temp1))

    def forward(self, x, train=True):
        for i, layer in enumerate(self.layers.values()):
            x = F.relu(layer(x))
            x0 = x
            if train == True:
                if i < self.num_layers_boosted:
                    self.boosted_layers[i] = torch.from_numpy(np.array(self.xg.fit(x0.detach().numpy(), (
                        self.l).detach().numpy()).feature_importances_) + self.epsilon)
        if self.ch == 1:
            x = torch.sigmoid(x)
        elif self.ch == 2:
            x = F.softmax(x)
        else:
            pass
        return x

