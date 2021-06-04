import torch
from collections import OrderedDict

class Seq(torch.nn.Sequential):
    def give(self, xg, num_layers_boosted, ep=0.001):
        self.xg = xg
        self.epsilon = ep
        self.boosted_layers = OrderedDict()
        self.num_layers_boosted = num_layers_boosted

    def forward(self, input, l,train):
        self.l = l
        for i, module in enumerate(self):
            input = module(input)
            x0 = input
            if train:
                if i < self.num_layers_boosted:
                    self.boosted_layers[i] = torch.from_numpy(np.array(
                        self.xg.fit(x0.detach().numpy(), (self.l).detach().numpy()).feature_importances_) + self.epsilon)
        return input