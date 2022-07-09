from flask import Flask, render_template, request, send_file
import csv
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from XBNet.training_utils import training, predict
from XBNet.models import XBNETClassifier
from XBNet.run import run_XBNET
import matplotlib.pyplot as plt
import os
import shutil
import pickle

app = Flask(__name__)


@app.route('/')
def uploady_file():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    global df
    global model_name
    global layers
    global target, n_layers_boosted
    if request.method == 'POST':
        df = pd.read_csv(request.files['csvfile'])
        model_name = request.form['model']
        layers = request.form['num_layers']
        target = request.form['target']

        model_name = model_name.lower()
        if len(layers) > 0:
            layers = int(layers)
        target = target
        if model_name.lower() == "xbnet":
            n_layers_boosted = 1
            layers = [i + 1 for i in range(int(layers))]
            process_input()
            return render_template('layers.html', layers=layers)
            # self.net_model()
        elif (model_name == "xgboost" or model_name == "randomforest"
              or model_name == "decision tree" or model_name == "lightgbm"):
            process_input()
            return render_template('treesinp.html', layers=layers)
            # self.tree_model()
        elif model_name.lower() == "neural network":
            n_layers_boosted = 0
            layers = [i + 1 for i in range(int(layers))]
            process_input()
            return render_template('layers.html', layers=layers)
            # self.net_model()

        # get number of layers, preferably produce a list

        # return 'file uploaded successfully'


@app.route('/layers', methods=['GET', 'POST'])
def getlayers():
    global layers_dims
    fileExt = r".pt"
    file = [_ for _ in os.listdir(os.getcwd()) if _.endswith(fileExt)]
    if len(file) > 0:
        os.remove(file[0])
    fileExt = r".pkl"
    file = [_ for _ in os.listdir(os.getcwd()) if _.endswith(fileExt)]
    if len(file) > 0:
        os.remove(file[0])
    layers_dims = []
    if request.method == 'POST':
        if model_name.lower() == "xbnet" or model_name.lower() == "neural network":
            for i in layers:
                layers_dims.append(int(request.form["i" + str(i)]))
                layers_dims.append(int(request.form["o" + str(i)]))
            print(layers_dims)
            train()
            path = os.path.join("static/", "images")
            print(path)
            if os.path.isdir(path) == False:
                os.mkdir(path)
            if os.path.isfile("static\images\Training_graphs.png") == True:
                os.remove("static\images\Training_graphs.png")
            shutil.move("Training_graphs.png", path)
            return render_template("results.html", info={"training_acc": acc[-1], "testing_acc": val_ac[-1],
                                                         "img": True})

        elif (model_name == "xgboost" or model_name == "randomforest"
              or model_name == "decision tree" or model_name == "lightgbm"):
            for i in request.form.keys():
                try:
                    layers_dims.append(int(request.form[i]))
                except:
                    layers_dims.append(float(request.form[i]))
            print(layers_dims)
            train()

            return render_template("results.html", info={"training_acc": training_acc*100,
                                                         "testing_acc": testing_acc*100, "img": False})


@app.route('/default', methods=['GET', 'POST'])
def default():
    global layers_dims
    global layers_dims
    fileExt = r".pt"
    file = [_ for _ in os.listdir(os.getcwd()) if _.endswith(fileExt)]
    if len(file) > 0:
        os.remove(file[0])
    fileExt = r".pkl"
    file = [_ for _ in os.listdir(os.getcwd()) if _.endswith(fileExt)]
    if len(file) > 0:
        os.remove(file[0])
    layers_dims = [100, 6, 0.3, 1, 1]
    train()
    return render_template("results.html", info={"training_acc": training_acc*100, "testing_acc": testing_acc*100,
                                                 "img": False})


@app.route('/download', methods=['GET', 'POST'])
def download():
    if model_name.lower() == "xbnet" or model_name.lower() == "neural network":
        fileExt = r".pt"
        file = [_ for _ in os.listdir(os.getcwd()) if _.endswith(fileExt)][0]
        return send_file(file)
    else:
        fileExt = r".pkl"
        file = [_ for _ in os.listdir(os.getcwd()) if _.endswith(fileExt)][0]
        return send_file(file)


def process_input():
    global x_data, y_data, label_encoded, imputations, label_y, columns_finally_used, y_label_encoder
    column_to_predict = target
    data = df
    n_df = len(data)
    label_encoded = {}
    imputations = {}
    for i in data.columns:
        imputations[i] = data[i].mode()
        if data[i].isnull().sum() / n_df >= 0.15:
            data.drop(i, axis=1, inplace=True)
        elif data[i].isnull().sum() / n_df < 0.15 and data[i].isnull().sum() / n_df > 0:
            data[i].fillna(data[i].mode(), inplace=True)
            imputations[i] = data[i].mode()
    columns_object = list(data.dtypes[data.dtypes == object].index)
    for i in columns_object:
        if i != column_to_predict:
            if data[i].nunique() / n_df < 0.4:
                le = LabelEncoder()
                data[i] = le.fit_transform(data[i])
                label_encoded[i] = le
            else:
                data.drop(i, axis=1, inplace=True)

    x_data = data.drop(column_to_predict, axis=1).to_numpy()
    columns_finally_used = data.drop(column_to_predict, axis=1).columns

    y_data = data[column_to_predict].to_numpy()
    label_y = False
    if y_data.dtype == object:
        label_y = True
        y_label_encoder = LabelEncoder()
        y_data = y_label_encoder.fit_transform(y_data)
    print("Number of features are: " + str(x_data.shape[1]) +
          " classes are: " + str(len(np.unique(y_data))))


def train():
    global model_tree, model_trained, acc, val_ac, training_acc, testing_acc
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=0.3, random_state=0)
    if model_name == "xbnet" or model_name == "neural network":
        m = model_name
        print(layers)
        print(layers_dims, n_layers_boosted)
        model = XBNETClassifier(X_train, y_train, num_layers=int(len(layers) / 2), num_layers_boosted=n_layers_boosted,
                                input_through_cmd=True, inputs_for_gui=layers_dims,
                                )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model_trained, acc, lo, val_ac, val_lo, cl = run_XBNET(X_train, X_test, y_train, y_test, model,
                                                               criterion, optimizer, 32, 300, save=True)
        print(type(cl))
        model_trained.save(m + "_testAccuracy_" + str(max(val_ac))[:4] + "_trainAccuracy_" +
                           str(max(acc))[:4] + ".pt", )
        # toast("Test Accuracy is: " +str(max(val_ac))[:4] +" and Training Accuracy is: " +
        #             str(max(acc))[:4] + " and model is saved.",duration= 10)
        return render_template("results.html", info={"training_acc": acc, "testing_acc": val_ac})

    elif (model_name == "xgboost" or model_name == "randomforest"
          or model_name == "decision tree" or model_name == "lightgbm"):
        if model_name == "xgboost":
            model_tree = XGBClassifier(n_estimators=layers_dims[0],
                                       max_depth=layers_dims[1],
                                       learning_rate=layers_dims[2],
                                       subsample=layers_dims[3],
                                       colsample_bylevel=layers_dims[4],
                                       random_state=0, n_jobs=-1,
                                       )
            model_tree.fit(X_train, y_train, eval_metric="mlogloss")
            training_acc = model_tree.score(X_train, y_train)
            testing_acc = model_tree.score(X_test, y_test)
        elif model_name == "randomforest":
            model_tree = RandomForestClassifier(n_estimators=layers_dims[0],
                                                max_depth=layers_dims[1],
                                                random_state=0, n_jobs=-1)
            model_tree.fit(X_train, y_train)
            training_acc = model_tree.score(X_train, y_train)
            testing_acc = model_tree.score(X_test, y_test)
        elif model_name == "decision tree":
            model_tree = DecisionTreeClassifier(max_depth=layers_dims[1], random_state=0)
            model_tree.fit(X_train, y_train)
            training_acc = model_tree.score(X_train, y_train)
            testing_acc = model_tree.score(X_test, y_test)
        elif model_name == "lightgbm":
            model_tree = LGBMClassifier(n_estimators=layers_dims[0],
                                        max_depth=layers_dims[1],
                                        learning_rate=layers_dims[2],
                                        subsample=layers_dims[3],
                                        colsample_bylevel=layers_dims[4],
                                        random_state=0, n_jobs=-1, )
            model_tree.fit(X_train, y_train, eval_metric="mlogloss")
            training_acc = model_tree.score(X_train, y_train)
            testing_acc = model_tree.score(X_test, y_test)
        print("Training and Testing accuracies are " + str(training_acc * 100)
              + " " + str(testing_acc * 100) + " respectively and model is stored")
        with open(model_name + "_testAccuracy_" + str(testing_acc)[:4] + "_trainAccuracy_" +
                  str(training_acc)[:4] + ".pkl", 'wb') as outfile:
            pickle.dump(model_tree, outfile)


@app.route('/predict', methods=['GET', 'POST'])
def predict_results():
    df_predict = pd.read_csv(request.files["csvpredictfile"])
    print(list(columns_finally_used))
    data = df_predict[list(columns_finally_used)]
    for i in data.columns:
        if data[i].isnull().sum() > 0:
            data[i].fillna(imputations[i], inplace=True)
        if i in label_encoded.keys():
            data[i] = label_encoded[i].transform(data[i])
    if (model_name == "xgboost" or model_name == "randomforest"
            or model_name == "decision tree" or model_name == "lightgbm"):
        predictions = model_tree.predict(data.to_numpy())
    else:
        predictions = predict(model_trained, data.to_numpy())
        if label_y == True:
            df_predict[target] = y_label_encoder.inverse_transform(predictions)
        else:
            df_predict[target] = predictions
    df_predict.to_csv("Predicted_Results.csv", index=False)
    return send_file("Predicted_Results.csv")
    # toast(text="Predicted_Results.csv in this directory has the results",
    #                    duration = 10)


if __name__ == '__main__':
    app.run(debug=True)