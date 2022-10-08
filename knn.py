import numpy as np
from typing import List, Union
from collections import Counter

class KNN:
    def __init__(self, k:int) -> None:
        self._k_neighbours = k
    
    def fit(self, Xtrain:Union[np.array, List], ytrain:Union[np.array, List]) -> None:
        # Save the dataset
        print(f'Fitting the model . . .')
        self._Xtrain = Xtrain
        self._ytrain = ytrain
        print(f'Fit completed')
    
    def predict(self, Xtest: Union[np.array, List]) -> np.array:
        return np.array([self._predict_one_sample(test_sample) for test_sample in Xtest])
    
    def _predict_one_sample(self, x: Union[np.array, List]) -> int:
        distances = [self._euclidean_distance(x, train_sample) for train_sample in self._Xtrain]
        k_indices = np.argsort(distances)[:self._k_neighbours]
        k_labels = [self._ytrain[i] for i in k_indices]
        highly_voted_label = self._majority_voting(k_labels)
        return highly_voted_label
    
    def _euclidean_distance(self, A:Union[np.array, List], B: Union[np.array, List]) -> float:
        # find euclidean distance between vectors A and B
        return np.sqrt(np.sum((A - B)**2))
    
    def _majority_voting(self, labels:Union[np.array, List]) -> int:
        return Counter(labels).most_common(1)[0][0]
        

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    print(f'Train Data Shape: {X_train.shape}\nTrain Label Shape: {y_train.shape}')
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    print(f'KNN Classifier F1-Score: {f1_score(y_test, predictions, average="weighted")}')
    