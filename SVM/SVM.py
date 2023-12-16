from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
import pandas as pd
import numpy as np
from collections import namedtuple

Dataset = namedtuple("Dataset", ["X", "y"])

class DatasetHelper:
    def read(self, dataset_path):
        data = pd.read_csv(dataset_path)

        # X-> Contains the features
        X = data.iloc[:, 0:-1]
        # y-> Contains all the targets
        y = data.iloc[:, -1]

        dataset = Dataset(X, y)
        return dataset


class SVM:

    def __init__(self, datasets):
        self.datasets = datasets
        self.models = []

    def train_model(self, model, dataset):
        if model:
            X = dataset.X
            y = dataset.y

            # Train the model
            model.fit(X, y)

    def build_models(self):
        model1 = SVR(kernel="rbf")
        model2 = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
        model3 = Pipeline([('scaler', StandardScaler()), ('svc', SVC(gamma=5,C=7))])
        self.models.extend([model1, model2, model3])
        assert len(self.models) == len(self.datasets), \
            f"Number of models {len(self.models)} is not the same as number of datasets {len(self.datasets)}"

        for i in range(len(self.models)):
            self.train_model(self.models[i], self.datasets[i])

