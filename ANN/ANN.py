import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix


def split_and_standardize(X,y):
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test=train_test_split(X,y)

    scaler = scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def create_model(X_train,y_train):
    model_1 = MLPClassifier((100,200,100),'tanh')
    model_2 = MLPClassifier((20,50,20),'logistic')
    
    

    
    model_1 = model_1.fit(X_train,y_train)
    model_2 = model_2.fit(X_train,y_train)
    
    return model_1,model_2

def predict_and_evaluate(model,X_test,y_test):
    
    y_predicted = model.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_predicted)
    precision = precision_score(y_test,y_predicted,average='micro')
    recall = recall_score(y_test,y_predicted,average='micro')
    f1 = f1_score(y_test,y_predicted,average='micro')
    confusion = confusion_matrix(y_test,y_predicted)

    
    return accuracy,precision,recall,f1,confusion




        

