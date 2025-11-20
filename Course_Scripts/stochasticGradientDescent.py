import numpy as np
import math
###################################################
# Optimization Problem

trueW = np.array([1, 2, 3, 4, 5])
def generate():
    x = np.random.rand(len(trueW)) #returns 5 values between 0 and 1. e.g 0.12, 0.89...
    #vectorial produc between trueW, x (data generated) and additional random number 
    #from a standard normal distribution. As noise for realism
    y = trueW.dot(x) + np.random.randn()
    #print('example:', x, y)
    return(x,y)

trainExamples = [generate() for i in range(1000000)]


def phi(x):
    return np.array(x)

def initialWeightVector():
    return np.zeros(len(trueW) )

def trainLoss(w):
    return 1.0 / len(trainExamples) * sum ((w.dot(phi(x)) - y)**2  for x, y in trainExamples)

def gradientTrainLoss(w):
    return 1.0 / len(trainExamples) * sum (2 * (w.dot(phi(x)) - y) * phi(x)  for x, y in trainExamples)

def loss(w,i):
    x,y = trainExamples[i]
    return (w.dot(phi(x)) - y)**2 

def gradientLoss(w,i):
    x,y = trainExamples[i]
    return 2 * (w.dot(phi(x)) - y) * phi(x)

####################################################
# Optimization Algorithm

def gradientDescent(F, gradientF, initialWeightVector):
    w = initialWeightVector()
    eta = 0.1
    for t in range(500):
        value = F(w)
        gradient = gradientF(w)
        w = w - eta * gradient
        print (f' epoch {t}: w = {w}, F{w} = {value}, gradientF = {gradient} ')
def stochasticGradientDescent(f, gradientf, n, initialWeightVector):
    w = initialWeightVector()
    numberUpdates = 0 
    n = len(trainExamples)
    for t in range(500):
        for i in range(n):
            value = f(w,i)
            #pick a random example
            i = np.random.randint(n)
            gradient = gradientf(w,i)
            numberUpdates += 1
            eta = 1.0 / math.sqrt(numberUpdates)
            w = w - eta * gradient
        print (f' epoch {t}: w = {w}, F{w} = {value}, gradientF = {gradient} ')
stochasticGradientDescent(loss, gradientLoss,len(trainExamples), initialWeightVector)

