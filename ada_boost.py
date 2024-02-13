import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class AdaBoost:
    def __init__(self, n_estimators=10) -> None:
        self.n_estimators = 10
        self.estimators = []

    def fit_boosting(self, X, y) -> None:
        n_samples, n_features = X.shape
        D = np.ones((n_samples,))

        for t in range(self.n_estimators):
            # Normalized training sample weight
            D = D/np.sum(D)

            # decision stumps
            h = DecisionTreeClassifier(max_depth=1)
            h.fit(X, y, sample_weight=D)

            y_hat = h.predict(X)

            # traning error (scaler)
            e = 1-accuracy_score(y, y_hat, sample_weight=D)

            # Model weight alpha (scaler)
            a = (1/2) * np.log((1-e)/e)

            # Convert True to 1 and False to -1
            m = (y == y_hat)*1 + (y != y_hat)*-1

            D *= np.exp(-a * m)

            self.estimators.append((a, h))

    def predict_boosting(self, X, y):

        y_hat = np.zeros((X.shape[0],))
        for a, h in self.estimators:
            # Positive values will become more positive
            # Negative values will become more negative
            y_hat += a*h.predict(X)

        # Change the values to sign 
        y_hat = np.sign(y_hat)

        print(f"Accuracy : {accuracy_score(y,y_hat)}")


if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=200, noise=0.1, random_state=0)
    # convert label to -1 & 1
    y = (2*y)-1
    Xtrn, Xtst, ytrn, ytst = train_test_split(
        X, y, test_size=0.25, random_state=0)

    model = AdaBoost()
    model.fit_boosting(Xtrn, ytrn)
    model.predict_boosting(Xtst, ytst)
