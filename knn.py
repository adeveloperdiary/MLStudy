import numpy as np
import seaborn as sns

class KNN:
    def __init__(self,k:int=3) -> None:
        self.k=k
        self.eps=1e-8
        
    
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    
    def _distance(self,X_test,X_train):
        def square(X):
            return np.sum(X**2, axis=1, keepdims=True)
    
        # (a-b)^2 = a^2 - 2a (dot) b.T + (b^2).T
        
        X_test_squared=square(X_test)
        X_train_squared=square(X_train)        
        
        X_test_dot_X_train=np.dot(X_test,X_train.T)

        return np.sqrt(self.eps+X_test_squared-2*X_test_dot_X_train+X_train_squared.T)
        
    def predict(self,X,y):
        distances=self._distance(X,self.X_train)
        
        y_hat=np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            y_idx=np.argsort(distances[i,:])
            k_closest_classes=self.y_train[y_idx[:self.k]]
            y_hat[i]=np.argmax(np.bincount(k_closest_classes))
    

        print(f"Acuracy: {self._accuracy(y,y_hat)}")
        return y_hat
        
    def _accuracy(self, y, y_hat):
        return np.sum(y == y_hat)/y.shape[0]
        
    
if __name__=="__main__":
    data=sns.load_dataset("iris")
    data=data[data['species']!='setosa']
    
    data['label']=0
    data.loc[data['species']=='versicolor','label']=1
    
    data = data.sample(frac=1).reset_index(drop=True)
    
    y = data.label.values
    X = data.drop(columns=['species', 'label']).values

    model = KNN()
    model.fit(X[:75, :], y[:75])
    model.predict(X[-25:, :], y[-25:])
    