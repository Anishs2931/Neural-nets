import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_data():
    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype(int)
    X = X / 255.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    return (X_train, y_train),(X_test, y_test)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# Data Wrapper Function
def load_data_wrapper():
    (X_train, y_train), (X_test, y_test) = load_data()
    
    training_inputs = [np.reshape(x, (784, 1)) for x in X_train]
    training_results = [vectorized_result(y) for y in y_train]
    training_data = list(zip(training_inputs, training_results))
       
    test_inputs = [np.reshape(x, (784, 1)) for x in X_test]
    test_data = list(zip(test_inputs, y_test))
    
    return (training_data, test_data)
