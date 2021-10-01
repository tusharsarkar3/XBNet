import torch
import numpy as np
from xgboost import XGBClassifier,XGBRegressor
from collections import OrderedDict
from XBNet.Seq import Seq

class XBNETClassifier(torch.nn.Module):
    '''
    XBNetClassifier is a model for classification tasks that tries to combine tree-based models with
    neural networks to create a robust architecture.
         :param X_values(numpy array): Features on which model has to be trained
         :param y_values(numpy array): Labels of the features i.e target variable
         :param num_layers(int): Number of layers in the neural network
         :param num_layers_boosted(int,optional): Number of layers to be boosted in the neural network. Default value: 1
         :param input_through_cmd(Boolean): Use to tell how you provide the inputs
         :param inputs_for_gui(list): Use only for providing inputs through list and when input_through_cmd is
                set to True
    '''
    def __init__(self, X_values, y_values, num_layers, num_layers_boosted=1,
                 input_through_cmd = False,inputs_for_gui=None):
        super(XBNETClassifier, self).__init__()
        self.name = "Classification"
        self.layers = OrderedDict()
        self.boosted_layers = {}
        self.num_layers = num_layers
        self.num_layers_boosted = num_layers_boosted
        self.X = X_values
        self.y = y_values
        self.gui = input_through_cmd
        self.inputs_layers_gui = inputs_for_gui

        self.take_layers_dim()
        self.base_tree()

        self.layers[str(0)].weight = torch.nn.Parameter(torch.from_numpy(self.temp.T))


        self.xg = XGBClassifier(n_estimators=100)

        self.sequential = Seq(self.layers)
        self.sequential.give(self.xg, self.num_layers_boosted)
        self.feature_importances_ = None

    def get(self, l):
        '''
        Gets the set of current actual outputs of the inputs
        :param l(tensor): Labels of the current set of inputs that are getting processed.
        '''
        self.l = l


    def take_layers_dim(self):
        '''
        Creates the neural network by taking input from the user
        :param gyi(Boolean): Is it being for GUI building purposes
        '''
        if self.gui == True:
            counter = 0
            for i in range(self.num_layers):
                inp = self.inputs_layers_gui[counter]
                counter += 1
                out = self.inputs_layers_gui[counter]
                counter += 1
                set_bias = True
                self.layers[str(i)] = torch.nn.Linear(inp, out, bias=set_bias)
                if i == 0:
                    self.input_out_dim = out
                self.labels = out
        else:
            print("Enter dimensions of linear layers: ")
            for i in range(self.num_layers):
                inp = int(input("Enter input dimensions of layer " + str(i + 1) + ": "))
                out = int(input("Enter output dimensions of layer " + str(i + 1)+ ": "))
                set_bias = bool(input("Set bias as True or False: "))
                self.layers[str(i)] = torch.nn.Linear(inp, out, bias=set_bias)
                if i == 0:
                    self.input_out_dim = out
                self.labels = out
            print("Enter your last layer ")
            self.ch = int(input("1. Sigmoid \n2. Softmax \n3. None \n"))
            if self.ch == 1:
                self.layers[str(self.num_layers)] = torch.nn.Sigmoid()
            elif self.ch == 2:
                dimension = int(input("Enter dimension for Softmax: "))
                self.layers[str(self.num_layers)] = torch.nn.Softmax(dim=dimension)
            else:
                pass

    def base_tree(self):
        '''
        Instantiates and trains a XGBRegressor on the first layer of the neural network to set its feature importances
         as the weights of the layer
        '''
        self.temp1 = XGBClassifier(n_estimators=100).fit(self.X, self.y,eval_metric="mlogloss").feature_importances_
        self.temp = self.temp1
        for i in range(1, self.input_out_dim):
            self.temp = np.column_stack((self.temp, self.temp1))

    def forward(self, x, train=True):
        x = self.sequential(x, self.l,train)
        return x

    def save(self,path):
        '''
        Saves the entire model in the provided path
        :param path(string): Path where model should be saved
        '''
        torch.save(self,path)


class XBNETRegressor(torch.nn.Module):
    '''
    XBNETRegressor is a model for regression tasks that tries to combine tree-based models with
    neural networks to create a robust architecture.
         :param X_values(numpy array): Features on which model has to be trained
         :param y_values(numpy array): Labels of the features i.e target variable
         :param num_layers(int): Number of layers in the neural network
         :param num_layers_boosted(int,optional): Number of layers to be boosted in the neural network. Default value: 1
    '''
    def __init__(self, X_values, y_values, num_layers, num_layers_boosted=1):
        super(XBNETRegressor, self).__init__()
        self.name = "Regression"
        self.layers = OrderedDict()
        self.boosted_layers = {}
        self.num_layers = num_layers
        self.num_layers_boosted = num_layers_boosted
        self.X = X_values
        self.y = y_values

        self.take_layers_dim()
        self.base_tree()

        self.layers[str(0)].weight = torch.nn.Parameter(torch.from_numpy(self.temp.T))


        self.xg = XGBRegressor(n_estimators=100)

        self.sequential = Seq(self.layers)
        self.sequential.give(self.xg, self.num_layers_boosted)
        self.sigmoid = torch.nn.Sigmoid()
        self.feature_importances_ = None

    def get(self, l):
        '''
        Gets the set of current actual outputs of the inputs
        :param l(tensor): Labels of the current set of inputs that are getting processed.
        '''
        self.l = l


    def take_layers_dim(self):
        '''
        Creates the neural network by taking input from the user
        '''
        print("Enter dimensions of linear layers: ")
        for i in range(self.num_layers):
            inp = int(input("Enter input dimensions of layer " + str(i + 1) + ": "))
            out = int(input("Enter output dimensions of layer " + str(i + 1)+ ": "))
            set_bias = bool(input("Set bias as True or False: "))
            self.layers[str(i)] = torch.nn.Linear(inp, out, bias=set_bias)
            if i == 0:
                self.input_out_dim = out
            self.labels = out

        print("Enter your last layer ")
        self.ch = int(input("1. Sigmoid \n2. Softmax \n3. None \n"))
        if self.ch == 1:
            self.layers[str(self.num_layers)] = torch.nn.Sigmoid()
        elif self.ch == 2:
            dimension = int(input("Enter dimension for Softmax: "))
            self.layers[str(self.num_layers)] = torch.nn.Softmax(dim=dimension)
        else:
            pass

    def base_tree(self):
        '''
        Instantiates and trains a XGBRegressor on the first layer of the neural network to set its feature importances
         as the weights of the layer
        '''
        self.temp1 = XGBRegressor(n_estimators=100).fit(self.X, self.y,eval_metric="mlogloss").feature_importances_
        self.temp = self.temp1
        for i in range(1, self.input_out_dim):
            self.temp = np.column_stack((self.temp, self.temp1))

    def forward(self, x, train=True):
        x = self.sequential(x,self.l,train)
        return x

    def save(self,path):
        '''
        Saves the entire model in the provided path
        :param path(string): Path where model should be saved
        '''
        torch.save(self,path)