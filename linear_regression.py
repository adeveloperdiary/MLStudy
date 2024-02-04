import numpy as np


class LinearRegression:
    def __init__(self, lr: float = 1e-3, epochs: int = 2000, debug: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.debug = debug

    def fit(self, X, y):
        # Crate the bias column
        bias = np.ones((X.shape[0], 1))
        # Add bias with the data
        X = np.append(bias, X, axis=1)

        w = np.zeros((X.shape[1], 1))

        for epoch in self.epochs:
            pass


if __name__ == "__main__":
    # Create X
    X1 = np.random.randn(500, 1)
    X2 = np.random.randn(500, 1)
    # Axis = 1 will add as a new column
    X = np.append(X1, X2, axis=1)

    # Generate Y using X
    y = X1*3+X2*2+np.random.normal(0, 1)

    model = LinearRegression(debug=True)
    model.fit(X, y)
