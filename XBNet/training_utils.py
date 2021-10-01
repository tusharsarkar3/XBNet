import numpy as np
from sklearn.metrics import classification_report,r2_score,mean_absolute_error,mean_squared_error,mean_squared_log_error
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def training(model,trainDataload,testDataload,criterion,optimizer,epochs = 100,save = False):
    '''
    Training function for training the model with the given data
    :param model(XBNET Classifier/Regressor): model to be trained
    :param trainDataload(object of DataLoader): DataLoader with training data
    :param testDataload(object of DataLoader): DataLoader with testing data
    :param criterion(object of loss function): Loss function to be used for training
    :param optimizer(object of Optimizer): Optimizer used for training
    :param epochs(int,optional): Number of epochs for training the model. Default value: 100
    :return:
    list of training accuracy, training loss, testing accuracy, testing loss for all the epochs
    '''
    accuracy = []
    lossing = []
    val_acc = []
    val_loss = []
    for epochs in tqdm(range(epochs),desc="Percentage training completed: "):
        running_loss = 0
        predictions = []
        act = []
        correct = 0
        total = 0
        loss = None
        for inp, out in trainDataload:
            try:
                if out.shape[0] >= 1:
                    out = torch.squeeze(out, 1)
            except:
                pass
            model.get(out.float())
            y_pred = model(inp.float())
            if model.labels == 1:
                loss = criterion(y_pred, out.view(-1, 1).float())
            else:
                loss = criterion(y_pred, out.long())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            for i, p in enumerate(model.parameters()):
                if i < model.num_layers_boosted:
                    l0 = torch.unsqueeze(model.sequential.boosted_layers[i], 1)
                    lMin = torch.min(p.grad)
                    lPower = torch.log(torch.abs(lMin))
                    if lMin != 0:
                        l0 = l0 * 10 ** lPower
                        p.grad += l0
                    else:
                        pass
                else:
                    pass
            outputs = model(inp.float(),train = False)
            predicted = outputs
            total += out.float().size(0)
            if model.name == "Regression":
                pass
            else:
                if model.labels == 1:
                    for i in range(len(predicted)):
                        if predicted[i] < torch.Tensor([0.5]):
                            predicted[i] = 0
                        else:
                            predicted[i] =1

                        if predicted[i].type(torch.LongTensor) == out[i]:
                            correct += 1
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == out.long()).sum().item()

            predictions.extend(predicted.detach().numpy())
            act.extend(out.detach().numpy())
        lossing.append(running_loss/len(trainDataload))
        if model.name == "Classification":
            accuracy.append(100 * correct / total)
            print("Training Loss after epoch {} is {} and Accuracy is {}".format(epochs + 1,
                                                                                 running_loss / len(trainDataload),
                                                                                 100 * correct / total))
        else:
            accuracy.append(100*r2_score(out.detach().numpy(),predicted.detach().numpy()))
            print("Training Loss after epoch {} is {} and Accuracy is {}".format(epochs+1,running_loss/len(trainDataload),accuracy[-1]))
        v_l,v_a = validate(model,testDataload,criterion,epochs)
        val_acc.extend(v_a)
        val_loss.extend(v_l)
    if model.name == "Classification":
        print(classification_report(np.array(act),np.array(predictions)))
    else:
        print("R_2 Score: ", r2_score(np.array(act),np.array(predictions)))
        print("Mean Absolute error Score: ", mean_absolute_error(np.array(act),np.array(predictions)))
        print("Mean Squared error Score: ", mean_squared_error(np.array(act),np.array(predictions)))
        print("Root Mean Squared error Score: ", np.sqrt(mean_squared_error(np.array(act),np.array(predictions))))
    validate(model,testDataload,criterion,epochs,True)

    model.feature_importances_ = torch.nn.Softmax(dim=0)(model.layers["0"].weight[1]).detach().numpy()

    figure, axis = plt.subplots(2)
    figure.suptitle('Performance of XBNET')

    axis[0].plot(accuracy, label="Training Accuracy")
    axis[0].plot(val_acc, label="Testing Accuracy")
    axis[0].set_xlabel('Epochs')
    axis[0].set_ylabel('Accuracy')
    axis[0].set_title("XBNet Accuracy ")
    axis[0].legend()


    axis[1].plot(lossing, label="Training Loss")
    axis[1].plot(val_loss, label="Testing Loss")
    axis[1].set_xlabel('Epochs')
    axis[1].set_ylabel('Loss value')
    axis[1].set_title("XBNet Loss")
    axis[1].legend()
    if save == True:
        plt.savefig("Training_graphs.png")
    else:
        plt.show()

    return accuracy,lossing,val_acc,val_loss


@torch.no_grad()
def validate(model,testDataload,criterion,epochs,last=False):
    '''
    Function for validating the training on testing/validation data.
    :param model(XBNET Classifier/Regressor): model to be trained
    :param testDataload(object of DataLoader): DataLoader with testing data
    :param criterion(object of loss function): Loss function to be used for training
    :param epochs(int,optional): Number of epochs for training the model. Default value: 100
    :param last(Boolean, optional): Checks if the current epoch is the last epoch. Default: False
    :return:
    list of validation loss,accuracy
    '''
    valid_loss = 0
    accuracy = []
    lossing = []
    predictions = []
    act = []
    correct = 0
    total = 0
    for inp, out in testDataload:
        model.get(out.float())
        y_pred = model(inp.float(), train=False)
        if model.labels == 1:
            loss = criterion(y_pred, out.view(-1, 1).float())
        else:
            loss = criterion(y_pred, out.long())
        valid_loss += loss
        total += out.float().size(0)
        predicted = y_pred
        if model.name == "Regression":
            pass
        else:
            if model.labels == 1:
                for i in range(len(y_pred)):
                    if y_pred[i] < torch.Tensor([0.5]):
                        y_pred[i] = 0
                    else:
                        y_pred[i] = 1
                    if y_pred[i].type(torch.LongTensor) == out[i]:
                        correct += 1
            else:
                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted == out.long()).sum().item()

        predictions.extend(predicted.detach().numpy())
        act.extend(out.detach().numpy())
    lossing.append(valid_loss / len(testDataload))
    if model.name == "Classification":
        accuracy.append(100 * correct / total)
    else:
        accuracy.append(100 * r2_score(np.array(act), np.array(predictions)))
    if last:
        if model.name == "Classification":
            print(classification_report(np.array(act), np.array(predictions)))
        else:
            print("R_2 Score: ", r2_score(np.array(act), np.array(predictions)))
            print("Mean Absolute error Score: ", mean_absolute_error(np.array(act), np.array(predictions)))
            print("Mean Squared error Score: ", mean_squared_error(np.array(act), np.array(predictions)))
            print("Root Mean Squared error Score: ", np.sqrt(mean_squared_error(np.array(act), np.array(predictions))))
    if model.name == "Classification":
        print("Validation Loss after epoch {} is {} and Accuracy is {}".format(epochs+1, valid_loss / len(testDataload),
                                                                               100 * correct / total))
    else:
        print("Validation Loss after epoch {} is {} and Accuracy is {}".format(epochs+1, valid_loss / len(testDataload),
                                                                               100*r2_score(np.array(act), np.array(predictions))))
    return lossing, accuracy

def predict(model,X):
    '''
    Predicts the output given the correct input data
    :param model(XBNET Classifier/Regressor): model to be trained
    :param X: Feature for which prediction is required
    :return:
    predicted value(int)
    '''
    X = torch.from_numpy(X)
    y_pred = model(X.float(), train=False)
    if model.name == "Classification":
        if model.labels == 1:
            if y_pred < torch.Tensor([0.5]):
                y_pred = 0
            else:
                y_pred = 1
        else:
            y_pred = np.argmax(y_pred.detach().numpy(),axis=1)
        return y_pred
    else:
        return y_pred.detach().numpy()[0]