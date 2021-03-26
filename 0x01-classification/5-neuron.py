#!/usr/bin/env python3
import numpy as np


class Neuron:
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

        
    @property
    def b(self):
        """getter for b attribute"""
        return self.__b

    @property
    def A(self):
        """getter for A attribute"""
        return self.__A

    @property
    def W(self):
        """getter for W attribute"""
        return self.__W
      
    def forward_prop(self, X):
       """forward propagation function"""
         Y = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Y))
        return self.__A
      
    def cost(self, Y, A):
        """cost of the model using logistic regression"""
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost
      
    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        self.forward_prop(X)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return prediction, cost
      
     def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        dz = A - Y
        dw = np.matmul(X, dz.T) / A.shape[1]
        db = np.sum(dz) / A.shape[1]
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db
        
        
      
        
