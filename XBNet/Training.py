import torch

def training(model,trainDataload,testDataload,criterion,optimizer,epochs = 10):
    for epochs in range(epochs):
        running_loss = 0
        for inp, out in trainDataload:
            model.get(out.float())
            y_pred = model(inp.float())
            loss = criterion(y_pred, out.view(-1, 1).float())
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for i, p in enumerate(model.parameters()):
                if i < model.num_layers_boosted:
                    l0 = torch.unsqueeze(model.boosted_layers[i], 1)
                    lMin = torch.min(p.grad)
                    lPower = torch.log(torch.abs(lMin))
                    if lMin != 0:
                        l0 = l0 * 10 ** lPower
                        p.grad += l0
                    else:
                        pass
                else:
                    pass
        print('Epoch', epochs+1, 'Loss:', running_loss/len(trainDataload))
        validate(model,testDataload,criterion,epochs+1)

@torch.no_grad()
def validate(model,testDataload,criterion,epochs):
    valid_loss = 0
    for inp,out in testDataload:
        model.get(out.float())
        y_pred = model(inp.float(),train = False)
        print(y_pred.shape, out.shape)
        loss = criterion(y_pred, out.view(-1,1).float())
        valid_loss += loss
    print('Epoch',epochs,'Loss:',valid_loss/len(testDataload))
validate()