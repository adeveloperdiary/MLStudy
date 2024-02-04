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

        self.w = np.zeros((X.shape[1], 1))

        for epoch in range(self.epochs):
            y_hat = np.dot(X, self.w)
            cost = (1 / X.shape[0]) * (np.sum(np.square(y_hat - y)))

            if epoch % 100 == 0:
                print(f"Cost after {epoch}: {cost}")

            self.w -= self.lr * (2 / X.shape[0]) * (np.dot(X.T, (y_hat - y)))

    def predict(self, X, y):
        bias = np.zeros((X.shape[0], 1))
        X = np.append(bias, X, axis=1)
        y_hat = np.dot(X, self.w)
        cost=(1/X.shape[0])*(np.sum(np.square(y_hat-y)))
        print(f"Test Cost is: {cost}")
        


if __name__ == "__main__":
    # Create X
    X1 = np.random.randn(500, 1)
    X2 = np.random.randn(500, 1)
    # Axis = 1 will add as a new column
    X = np.append(X1, X2, axis=1)

    # Generate Y using X
    y = X1 * 3 + X2 * 2 + np.random.normal(0, 1)

    model = LinearRegression(debug=True)
    model.fit(X[:450,:], y[:450,:])
    
    model.predict(X[:-50,:], y[:-50,:])
