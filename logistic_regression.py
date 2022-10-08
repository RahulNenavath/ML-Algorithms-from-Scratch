import numpy as np
from typing import Union

class LogisticRegression:
    def __init__(self, epochs:int, learning_rate:float = 0.001) -> None:
        self._lr = learning_rate
        self._epochs = epochs
        self._weights = None
        self._bias = 1
        self._num_features = None
        self._num_samples = None
    
    def fit(self, Xtrain:Union[list, np.array], ytrain: Union[list, np.array]):
        self._num_samples, self._num_features = Xtrain.shape
        self._weights = np.zeros(self._num_features)
        
        for epoch in range(self._epochs):
            y_pred = self._sigmoid(np.dot(Xtrain, self._weights) + self._bias)
            loss = y_pred - ytrain
            
            # weight & bias gradients
            dw = np.dot(Xtrain.T, loss) * (1/self._num_samples)
            db = np.sum(loss) * (1/self._num_samples)
            
            # update weight & bias
            self._weights -= self._lr * dw
            self._bias -= self._lr * db
            
            print(f'Epoch: {epoch} completed')
    
    def predict(self, Xtest: Union[list, np.array]) -> np.array:
        output = self._sigmoid(np.dot(Xtest, self._weights) + self._bias)
        return np.array([1 if sample_result > 0.5 else 0 for sample_result in output])
    
    def _sigmoid(self, x:float) -> float:
        return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MinMaxScaler
    
    iris = datasets.load_breast_cancer()
    X, y = iris.data, iris.target    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    
    print(f'Train Data Shape: {X_train.shape}\nTrain Label Shape: {y_train.shape}')
    
    lr = LogisticRegression(epochs=5500, learning_rate=0.001)
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    
    print(f'KNN Classifier F1-Score: {f1_score(y_test, predictions, average="weighted")}')