import numpy as np
from typing import Union

class LinearRegression:
    def __init__(self, epochs:int, learning_rate:float=0.001) -> None:
        self._epochs = epochs
        self._lr = learning_rate
        self._num_samples = None
        self._num_features = None
        self._weights = None
        self._bias = 1
    
    def fit(self, Xtrain:Union[list, np.array], ytrain:Union[list, np.array]) -> None:
        self._num_samples, self._num_features = Xtrain.shape
        self._weights = np.zeros(self._num_features)
        
        # Gradient Desent
        
        for epoch in range(self._epochs):
            # Y predicted
            y_preds = np.dot(Xtrain, self._weights) + self._bias
            # Loss
            loss = y_preds - ytrain
            
            # weight & bias gradient
            dw = np.dot(Xtrain.T, loss) * (1/self._num_samples)
            db = np.sum(loss) * (1/self._num_samples)
            
            # weight & bias update
            self._weights -= self._lr * dw
            self._bias -= self._lr * db
            
            print(f'Epoch: {epoch} completed')
    
    def predict(self, Xtest:Union[list, np.array]) -> np.array:
        return np.dot(Xtest, self._weights) + self._bias

if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import MinMaxScaler
    
    boston = fetch_california_housing()
    X,y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    
    print(f'X Train shape: {X_train.shape}, y Train shape: {y_train.shape}')
    
    lr = LinearRegression(epochs=15_000, learning_rate=0.01)
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    
    print(f'LR R2 Scores: {r2_score(y_test, predictions)}')
    