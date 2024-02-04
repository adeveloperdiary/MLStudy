from sklearn.datasets import make_blobs
import numpy as np


class LogisticRegression:

    def __init__(self, lr: float = 1e-3, epochs: int = 2000, debug: bool = False) -> None:
        self.lr = lr
        self.epochs = epochs
        self.debug = debug

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        self.w = np.zeros((X.shape[1], 1))
        self.b = 0

        for epoch in range(self.epochs):
            y_hat = self.sigmoid(np.dot(X, self.w)+self.b)

            # Cross Entropy Loss
            cost = (-1/X.shape[1]) * \
                np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))

            # gradient calc
            self.w -= self.lr*(1/X.shape[1]) * np.dot(X.T, (y_hat-y))
            self.b -= self.lr*(1/X.shape[1]) * np.sum(y_hat-y)

            if epoch % 100 == 0:
                print(f"Cost after {epoch}: {cost}")

    def predict(self, X, y):
        y_hat = self.sigmoid(np.dot(X, self.w)+self.b)
        y_hat = y_hat > 0.5
        
        print(f"Accuracy: {np.sum(y==y_hat)/X.shape[0]}")


if __name__ == "__main__":
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=2)
    y = y.reshape(-1, 1)
    model = LogisticRegression(debug=True)
    model.fit(X, y)
    model.predict(X,y)
