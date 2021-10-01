from kivymd.app import MDApp
from kivy.uix.widget import Widget
from kivy.uix.actionbar import ActionBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivymd.theming import ThemableBehavior
from kivymd.uix.list import OneLineListItem, MDList, TwoLineListItem, ThreeLineListItem
from kivymd.uix.list import MDList
from kivymd.uix.textfield import MDTextField
from kivy.uix.button import Button
from kivy.lang import Builder
from kivymd.toast import toast
from kivy.uix.screenmanager import Screen, ScreenManager
import time
from kivy.core.window import Window
from kivymd.uix.label import MDLabel
from kivy.uix.modalview import ModalView
from kivymd.uix.filemanager import MDFileManager
from kivymd.theming import ThemeManager
import requests
from kivy.uix.popup import Popup
import os
from xgboost import XGBClassifier
from  sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import  DecisionTreeClassifier
from lightgbm import  LGBMClassifier
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from XBNet.training_utils import training,predict
from XBNet.models import XBNETClassifier
from XBNet.run import run_XBNET
from os import environ
import pickle

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

Login_Page = """
ScreenManager:
    LoginPage
    ModelDetails
    FileManage

<LoginPage>:
    name:"Login"
    MDFloatLayout:
        Image:
            id: imageView
            source: 'Untitled.png'
            allow_stretch: True
            halign: 'center'
            pos_hint: {"center_x":0.23, "center_y":0.5}

        MDRoundFlatIconButton:
            id: filemanage
            text: "Select Dataset"
            icon: "folder"
            pos_hint: {'center_x': .77, 'center_y': .85}
            on_release: root.manager.current = "File"
            
            
        MDTextField:
            id: modelname
            hint_text:"Enter the model name: "
            pos_hint:{"center_x":0.77,"center_y":0.7}
            current_hint_text_color:0,0,0,1
            size_hint_x:0.4
            required: True
            
        MDTextField:
            id: layers
            hint_text:"Enter number of layers(For XBNet or NN): "
            pos_hint:{"center_x":0.77,"center_y":0.55}
            current_hint_text_color:0,0,0,1
            size_hint_x:0.4
            
        MDTextField:
            id: target
            hint_text:"Enter name of target feature: "
            pos_hint:{"center_x":0.77,"center_y":0.40}
            current_hint_text_color:0,0,0,1
            size_hint_x:0.4
            required: True

        MDRaisedButton:
            text:"Build model"
            pos_hint:{"center_x":0.77,"center_y":0.25}
            size_hint_x:0.3
            on_release: root.manager.current = "Model"
            on_press: app.get_model(modelname.text,target.text,layers.text)
            theme_text_color:"Custom"
            text_color:0,0,0,1
            

<ModelDetails>:
    name:"Model"
    MDFloatLayout:    
        Image:
            id: imageView
            source: 'Untitled.png'
            allow_stretch: True
            halign: 'center'
            pos_hint: {"center_x":0.23, "center_y":0.5}    
            
        MDRaisedButton:
            text:"Train"
            pos_hint:{"center_x":0.63,"center_y":0.15}
            size_hint_x:0.2
            # on_release: root.manager.current = "Model"
            on_press: app.get_layers()
            theme_text_color:"Custom"
            text_color:0,0,0,1
            
        MDRaisedButton:
            text:"Predict"
            pos_hint:{"center_x":0.88,"center_y":0.15}
            size_hint_x:0.2
            # on_release: root.manager.current = "Model"
            on_press: app.predict()
            theme_text_color:"Custom"
            text_color:0,0,0,1
    
    
<FileManage>:
    name:"File"
    BoxLayout:   
        FileChooserListView:
            canvas.before:
                Color:
                    rgb: 0.1, 0.2, 0.5
                Rectangle:
                    pos: self.pos
                    size: self.size
            on_selection: app.get_path(*args)    
    
            """

class LoginPage(Screen):
    pass

class ModelDetails(Screen):
    pass

class CustomDropDown(BoxLayout):
    pass

class FileManage(Screen):
    pass

sm = ScreenManager()
sm.add_widget(LoginPage(name="Login"))
sm.add_widget(ModelDetails(name="Model"))
sm.add_widget(FileManage(name="File"))

class XBNetGUI(MDApp):

    def __init__(self):
        super(XBNetGUI, self).__init__()
        self.predict_phase = False

    class ContentNavigationDrawer(BoxLayout):
        pass

    class DrawerList(ThemableBehavior, MDList):
        pass

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        login_page = Builder.load_string(Login_Page)

        return login_page

    def get_layers(self):
        self.layers_dims = []
        if self.model == "xbnet" or self.model == "neural network":
            for i,j in self.fields.items():
                self.layers_dims.append(int(j.text))
                print(j.text)
        elif (self.model == "xgboost" or self.model == "randomforest"
            or self.model == "decision tree" or self.model == "lightgbm"):
            for i,j in self.fields.items():
                try:
                    self.layers_dims.append(int(j.text))
                except:
                    self.layers_dims.append(float(j.text))

        self.train()

    def process_input(self):
        suppress_qt_warnings()
        column_to_predict = self.target
        data = pd.read_csv(self.file_selected)
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
        self.columns_finally_used = data.drop(column_to_predict, axis=1).columns

        y_data = data[column_to_predict].to_numpy()
        self.label_y = False
        if y_data.dtype == object:
            self.label_y = True
            self.y_label_encoder = LabelEncoder()
            y_data = self.y_label_encoder.fit_transform(y_data)
        self.label_encoded = label_encoded
        self.imputations = imputations
        toast("Number of features are: " + str(x_data.shape[1]) +
                                " classes are: "+ str(len(np.unique(y_data))),duration=5)
        self.x_data = x_data
        self.y_data = y_data

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.x_data, self.y_data,
                                                            test_size=0.3, random_state=0)
        if self.model == "xbnet" or self.model =="neural network":
            print(self.layers_dims)
            m = self.model
            model = XBNETClassifier( X_train, y_train, self.layers,
                                    input_through_cmd=True, inputs_for_gui=self.layers_dims,
                                     num_layers_boosted=self.n_layers_boosted
                                    )
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            self.model, self.acc, self.lo, self.val_ac, self.val_lo = run_XBNET(X_train, X_test, y_train, y_test, model, criterion, optimizer, 32, 10)
            model.save(m+"_testAccuracy_" +str(max(self.val_ac))[:4] +"_trainAccuracy_" +
                                        str(max(self.acc))[:4]+ ".pt",)
            toast("Test Accuracy is: " +str(max(self.val_ac))[:4] +" and Training Accuracy is: " +
                        str(max(self.acc))[:4] + " and model is saved.",duration= 10)

        elif (self.model == "xgboost" or self.model == "randomforest"
            or self.model == "decision tree" or self.model == "lightgbm"):
            if self.model == "xgboost":
                self.model_tree = XGBClassifier(n_estimators=self.layers_dims[0],
                                      max_depth=self.layers_dims[1],
                                      learning_rate= self.layers_dims[2],
                                      subsample= self.layers_dims[3],
                                      colsample_bylevel = self.layers_dims[4],
                                      random_state=0,n_jobs=-1,
                                      )
                self.model_tree.fit(X_train, y_train,eval_metric="mlogloss")
                training_acc = self.model_tree.score(X_train, y_train)
                testing_acc = self.model_tree.score(X_test,y_test)
            elif self.model == "randomforest":
                self.model_tree = RandomForestClassifier(n_estimators=self.layers_dims[0],
                                               max_depth=self.layers_dims[1],
                                               random_state=0,n_jobs=-1)
                self.model_tree.fit(X_train, y_train)
                training_acc = self.model_tree.score(X_train, y_train)
                testing_acc = self.model_tree.score(X_test,y_test)
            elif self.model == "decision tree":
                self.model_tree = DecisionTreeClassifier(max_depth=self.layers_dims[1],random_state=0)
                self.model_tree.fit(X_train, y_train)
                training_acc = self.model_tree.score(X_train, y_train)
                testing_acc = self.model_tree.score(X_test,y_test)
            elif self.model == "lightgbm":
                self.model_tree = LGBMClassifier(n_estimators=self.layers_dims[0],
                                      max_depth=self.layers_dims[1],
                                      learning_rate= self.layers_dims[2],
                                      subsample= self.layers_dims[3],
                                      colsample_bylevel = self.layers_dims[4],
                                      random_state=0,n_jobs=-1,)
                self.model_tree.fit(X_train, y_train,eval_metric="mlogloss")
                training_acc = self.model_tree.score(X_train, y_train)
                testing_acc = self.model_tree.score(X_test,y_test)
            toast(text="Training and Testing accuracies are "+str(training_acc*100)
                           +" "+str(testing_acc*100) + " respectively and model is stored",duration=7)
            with open(self.model+"_testAccuracy_" +str(testing_acc)[:4] +"_trainAccuracy_" +
                                        str(training_acc)[:4]+ ".pkl", 'wb') as outfile:
                pickle.dump(self.model_tree,outfile)

    def predict(self):
        self.predict_phase = True
        self.root.current = "File"

    def predict_results(self):
        df = pd.read_csv(self.file_selected)
        data = df[self.columns_finally_used]
        for i in data.columns:
            if data[i].isnull().sum() > 0:
                data[i].fillna(self.imputations[i], inplace=True)
            if i in self.label_encoded.keys():
                data[i] = self.label_encoded[i].transform(data[i])
        if (self.model == "xgboost" or self.model == "randomforest"
            or self.model == "decision tree" or self.model == "lightgbm"):
            predictions = self.model_tree.predict(data.to_numpy())
        else:
            predictions = predict(self.model, data.to_numpy())
            if self.label_y == True:
                df[self.target] = self.y_label_encoder.inverse_transform(predictions)
            else:
                df[self.target] = predictions
        df.to_csv("Predicted_Results.csv",index=False)
        toast(text="Predicted_Results.csv in this directory has the results",
                           duration = 10)


    def get_model(self,model,target,layers):
        self.model = model.lower()
        if len(layers) > 0:
            self.layers = int(layers)
        self.target = target
        if self.model.lower() == "xbnet":
            self.n_layers_boosted = 1
            self.net_model()
        elif (self.model == "xgboost" or self.model == "randomforest"
            or self.model == "decision tree" or self.model == "lightgbm"):
            self.tree_model()
        elif self.model.lower() == "neural network":
            self.n_layers_boosted = 0
            self.net_model()

        self.process_input()

    def net_model(self):
        layout = self.root.get_screen('Model')
        gap = 1/(2*self.layers+2)
        counter = 1
        self.fields = {}
        for i in range(self.layers):
                lab1 = MDTextField(hint_text="Enter input dimensions of layer "+ str(i+1) +":",
                                   pos_hint={"center_x":0.77,"center_y":1-gap*(counter)},
                                    size_hint_x=.4, current_hint_text_color=[0,0,0,1] )

                counter+=1
                lab2 = MDTextField(hint_text="Enter output dimensions of layer "+ str(i+1) +":",
                                   pos_hint={"center_x":0.77,"center_y":1-gap*(counter)},
                                   size_hint_x=.4, current_hint_text_color=[0,0,0,1] )
                counter +=1
                layout.add_widget(lab1)
                layout.add_widget(lab2)
                self.fields["input_"+str(i+1)] = lab1
                self.fields["output_" + str(i+1)] = lab2

    def tree_model(self):
        layout = self.root.get_screen('Model')
        self.fields = {}
        lab1 = MDTextField(hint_text="Enter number of estimators: ",
                           pos_hint={"center_x":0.77,"center_y":0.85},
                            size_hint_x=.4, current_hint_text_color=[0,0,0,1] )

        lab2 = MDTextField(hint_text="Enter depth of trees[default:6](Typical 3-10): ",
                           pos_hint={"center_x":0.77,"center_y":0.7},
                           size_hint_x=.4, current_hint_text_color=[0,0,0,1] )

        lab3 = MDTextField(hint_text="Enter learning rate forr XGBoost(eta)[default:0.3]: ",
                           pos_hint={"center_x":0.77,"center_y":0.55},
                            size_hint_x=.4, current_hint_text_color=[0,0,0,1] )

        lab4 = MDTextField(hint_text="Enter size of subsample[default:1](Typical 0.5-1): ",
                           pos_hint={"center_x":0.77,"center_y":0.4},
                            size_hint_x=.4, current_hint_text_color=[0,0,0,1] )

        lab5 = MDTextField(hint_text="Enter size of colsample_bytree[default:1](Typical 0.5-1): ",
                           pos_hint={"center_x":0.77,"center_y":0.25},
                            size_hint_x=.4, current_hint_text_color=[0,0,0,1] )

        layout.add_widget(lab1)
        layout.add_widget(lab2)
        layout.add_widget(lab3)
        layout.add_widget(lab4)
        layout.add_widget(lab5)
        self.fields["no_trees"] = lab1
        self.fields["depth"] = lab2
        self.fields["learning_rate"] = lab3
        self.fields["subsample"] = lab4
        self.fields["colsample_bytree"] = lab5

    def get_path(self,*args):
        print(args)
        self.file_selected = args[1][0]
        print(self.file_selected)
        if self.predict_phase:
            self.root.current = "Model"
            print("hellooo")
            self.predict_results()
        else:
            self.root.current = "Login"

if __name__ == "__main__":
    XBNetGUI().run()