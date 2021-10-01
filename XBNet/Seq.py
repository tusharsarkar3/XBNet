import torch
import numpy as np
from collections import OrderedDict

class Seq(torch.nn.Sequential):
    '''
     Seq uses sequential module to implement tree in the forward.
    '''
    def give(self, xg, num_layers_boosted, ep=0.001):
        '''
        Saves various information into the object for further usage in the training process
        :param xg(object of XGBoostClassifier): Object og XGBoostClassifier
        :param num_layers_boosted(int,optional): Number of layers to be boosted in the neural network.
        :param ep(int,optional): Epsilon for smoothing. Deafult: 0.001
        '''
        self.xg = xg
        self.epsilon = ep
        self.boosted_layers = OrderedDict()
        self.num_layers_boosted = num_layers_boosted

    def forward(self, input,train,l=torch.Tensor([1])):
        l,train = train,l
        for i, module in enumerate(self):
            input = module(input)
            x0 = input
            if train:
                self.l = l
                if i < self.num_layers_boosted:
                    try:
                        self.boosted_layers[i] = torch.from_numpy(np.array(
                            self.xg.fit(x0.detach().numpy(), (self.l).detach().numpy(),eval_metric="mlogloss").feature_importances_) + self.epsilon)
                    except:
                        pass
        return input