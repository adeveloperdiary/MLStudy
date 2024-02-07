import numpy as np
import seaborn as sns
import pandas as pd
import math
from scipy.stats import multivariate_normal

class NaiveBayes():

    def __init__(self) -> None:
        self.eps = 1e-6

    def _normalize(self, X):
        if self.mean is None and self.std is None:
            # Calculate mean and std for normalizing data
            self.mean = np.mean(X, axis=0)  # row ?
            self.std = np.std(X, axis=0)

        # normalize
        X_scaled = (X-self.mean)/self.std
        return X_scaled

    def fit(self, X, y) -> None:
        self.mean = None
        self.std = None
        X_scaled = self._normalize(X)

        self.num_classes = len(np.unique(y))

        self.class_mean = {}
        self.class_std = {}
        self.class_prior = {}

        for c in range(self.num_classes):
            X_temp = X_scaled[y == c]

            self.class_mean[c] = np.mean(X_temp, axis=0)  # per column
            self.class_std[c] = np.std(X_temp, axis=0)
            self.class_prior[c] = X_temp.shape[0]/X.shape[0]

    def _normal_pdf(self, X, mean, std):
        # pdf in log scale
        # pdf= -1/2 sum ( (X-Mean)**2/(std+eps) by row ) - 1/2 sum log (sigma+eps) - n_features/2 (log 2*pi)

        pdf1= - (1/2)*np.sum(np.square(X-mean)/(std+self.eps), axis=1) - \
            (1/2)*np.sum(np.log(std+self.eps)) - (X.shape[1]/2)*np.log(2*np.pi)
        
        pdf2= np.log(multivariate_normal.pdf(X,mean,std))
        
        return pdf2

    def predict(self, X, y):
        X_scaled = self._normalize(X)
        probs = np.zeros((X_scaled.shape[0], self.num_classes))

        for c in range(self.num_classes):
            class_prob = self._normal_pdf(
                X_scaled, self.class_mean[c], self.class_std[c])
            probs[:, c] = class_prob+self.class_prior[c]

        y_hat = np.argmax(probs, axis=1)
        print(f"Acuracy: {self._accuracy(y,y_hat)}")

        return y_hat

    def _accuracy(self, y, y_hat):
        return np.sum(y == y_hat)/y.shape[0]


if __name__ == "__main__":
    data = sns.load_dataset("iris")
    data = data[data['species'] != 'setosa']

    data['label'] = 0
    data.loc[data['species'] == 'versicolor', 'label'] = 1

    data = data.sample(frac=1).reset_index(drop=True)

    y = data.label.values
    X = data.drop(columns=['species', 'label']).values

    model = NaiveBayes()
    model.fit(X[:75, :], y[:75])
    model.predict(X[-25:, :], y[-25:])
