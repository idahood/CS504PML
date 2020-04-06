#!/usr/bin/env python3

# Homework 1: Solution
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

# Load data and data standardization
def load_data():
    '''Load the MNIST dataset'''

    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    return (X_train, y_train, X_test, y_test)

def data_std(X_train, X_test):
    '''Data standardization

    Parameters
    ----------
    X_train: origianl training set
    X_test: original test set

    Returns
    -------
    X_train_std: rescaled training set
    X_test_std: rescaled test set
    '''
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, X_test_std

def create_NN(n_features = 784, n_outputs = 10): # 30 points
    '''create a deep feedforward neural network using keras

    Parameters
    -----------
    n_features: the number of input features/units
    n_output: the number of output units

    Returns
    -------
    myNN: the neural network model

    '''
    ## add your code here
    myNN = Sequential()
    myNN.add(Dense(32, activation='relu', input_dim=n_features))
    myNN.add(Dense(64, activation='relu'))
    myNN.add(Dense(64, activation='relu'))
    myNN.add(Dense(n_outputs, activation='softmax'))
    myNN.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    ##

    return myNN

def nn_params_search(nn, X_train, y_train): # 30 points
    '''Search best paramaters for svm classifier

    Parameters
    ----------
    X_train: features
    y_train: target of the input


    Returns
    -------
    best_params_

    Example grid: (you can customize the search graid by youself)
    param_grid = [{'batch_size': [64, 128], 'epochs' : [10, 30, 50]}]

    '''
    ## add your code here
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    # https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
    # https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
    from sklearn.model_selection import GridSearchCV
    from sklearn import svm

    param_grid = [{'batch_size': [64, 128], 'epochs' : [10, 30, 50]}]
    grid_search = GridSearchCV(estimator=nn, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    grid_search.best_params_

    return grid_search.best_params_
    ##

def retrain_best_nn(best_params, X_train, y_train): # 10 points
    '''
    Retrain a svm classifier using the best parameters

    Paramters
    ----------
    best_params:
    X_train: data input of the training set
    y_train: target of the input

    Returns
    ---------
    bestNN: the nn classifier trained using the best parameters

    '''
    ## add your code here
    bestNN = create_NN()
    bestNN.fit(X_train,
               y_train,
               validation_split=0.2,
               verbose=1,
               batch_size = best_params['batch_size'],
               epochs = best_params['epochs'])
    ##

    return bestNN

def performance_acc(y, y_pred): # 10 points
    ''' calculate the confusion matrix and average accuracy

        Parameters
        ----------
        y: real target
        y_pred: prediction

        Returns
        -------
        cm: confusion matrix
        acc: accuracy
    '''
    ## add your code here
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    ##

    return cm, acc

from keras.wrappers.scikit_learn import KerasClassifier

if __name__ == '__main__':

    #Task 1. load the dataset
    (X_train, y_train, X_test, y_test) = load_data()

    # 1.1 reshape the training and test sets to N * 784. 10 points
    ## add your code here
    X_train_1 = X_train.reshape(X_train.shape[0], 784)
    X_test_1 = X_test.reshape(X_test.shape[0], 784)
    print('1. X_train: {}, X_train_1: {}'.format(X_train.shape, X_train_1.shape))
    ##

    # 1.2 transform y_train to one-hot vectors using keras.utils.to_categorical. 10 points
    ## add your code here
    num_classes = 10
    y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
    print('y_train_onehot: {}'.format(y_train_onehot.shape))
    ##

    #Task 2. create a deep feedforward neural network
    myNN = create_NN(X_train_1.shape[1], y_train_onehot.shape[1])
    myNN.summary()
    myNN1 = KerasClassifier(build_fn = create_NN, batch_size = 64, epochs = 50)

    #Task 3. Search best parameters, and report the performance
    best_params = nn_params_search(myNN1, X_train_1, y_train)
    print('Best parameters: ', best_params)

    bestNN = retrain_best_nn(best_params, X_train_1, y_train_onehot)
    y_test_pred = bestNN.predict_classes(X_test_1)
    cm, acc = performance_acc(y_test, y_test_pred)
    print('Confusion matrix:\n', cm)
    print('Accuracy =    {:.3f}%'.format(acc*100))

    #Task 4. Search best nn parameters after data standardization, and report the performance
    X_train_std, X_test_std = data_std(X_train_1, X_test_1)

    best_params =  nn_params_search(myNN1, X_train_std, y_train)
    print('Best parameters: ', best_params)

    bestNN_std = retrain_best_nn(best_params, X_train_std, y_train_onehot)
    y_test_std_pred = bestNN_std.predict_classes(X_test_std)
    cm1, acc1 = performance_acc(y_test, y_test_std_pred)
    print('Confusion matrix:\n', cm1)
    print('Accuracy =    {:.3f}%'.format(acc1*100))
