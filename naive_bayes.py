import numpy as np
from typing import List, Union

class NaiveBayes:
    def __init__(self) -> None:
        self._classes = None
        self._num_classes = None
        self._num_samples = None
        self._num_features = None
        self._class_means = None
        self._class_var = None
        self._class_priors = None
        pass
    
    def fit(self, Xtrain:Union[np.array, List], ytrain: Union[np.array, List]) -> None:
        # Calculate Mean, Variance, & Prior probabily for each target class 
        self._num_samples, self._num_features = Xtrain.shape
        self._classes = np.unique(ytrain)
        self._num_classes = len(self._classes)
        
        self._class_means = np.zeros((self._num_classes, self._num_features), dtype=np.float32)
        self._class_var = np.zeros((self._num_classes, self._num_features), dtype=np.float32)
        self._class_priors = np.zeros(self._num_classes, dtype=np.float32)
        
        for target_class in self._classes:
            X_class = Xtrain[target_class == ytrain]
            # Mean of each features per class
            self._class_means[target_class, : ] = np.mean(X_class, axis=0)
            # Variance of each features per class
            self._class_var[target_class, : ] = np.var(X_class, axis=0)
            # Class Prior Probabilites -> Class frequencies: class samples / total samples
            self._class_priors[target_class] = len(X_class) / self._num_samples

    
    def predict(self, Xtest:Union[np.array, List]) -> np.array:
        return np.array([self._predict_one_sample(x) for x in Xtest])
    
    def _predict_one_sample(self, x:Union[np.array, List]) -> int:
        # move through each class and find its posterior probability
        # Posterior Probs: P( y | X) = Class Conditional Probs + Class Prior Prob
        # Class Conditional Probs P(x_k / y_i) => Guassian PDF
        # Prior Probs P(y) => num samples in i-th class / total samples in dataset
        
        posteriers = []
        
        for index in range(self._num_classes):
            # formula for i-th class: P(y | X) = log(P(x_1 | y_i)) + log(P(x_2 | y_i)) + log(P(x_3 | y_i)) + . . . log(P(x_n | y_i)) + log(P(y_i))
            posterior_prob = np.sum(np.log(self._guassian_pdf(index, x)) + np.log(self._class_priors[index]))
            posteriers.append(posterior_prob)
        
        return self._classes[np.argmax(posteriers)]
    
    def _guassian_pdf(self, class_index, x) -> float:
        class_means = self._class_means[class_index]
        class_vars = self._class_var[class_index]
        
        numerator = np.exp(-( (x - class_means)**2/ (2 * (class_vars)**2)))
        denominator = np.sqrt(2* np.pi * (class_vars)**2)
        
        return numerator / denominator

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MinMaxScaler
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    
    print(f'Train Data Shape: {X_train.shape}\nTrain Label Shape: {y_train.shape}')
    print(f'Train Labels: {np.unique(y_train)}')
    
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    
    print(f'Naive Bayes Classifier F1-Score: {f1_score(y_test, predictions, average="weighted")}')