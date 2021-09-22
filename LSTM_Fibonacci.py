from fibonacci_class import Fibonacci
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from matplotlib import pyplot

fibonacci= Fibonacci()
data= [fibonacci(n) for n in range(20)] # create the Fibonacci sequence
print(data)

# Split the data using the following function
def splitSequence(seq, n_steps):
    X= []
    y= []
    for i in range(len(seq)): #get the last index
        lastIndex= i+ n_steps
        if lastIndex> len(seq)- 1: #if lastIndex is greater than length of sequence then break
            break
        seq_X, seq_y= seq[i:lastIndex], seq[lastIndex] #Create input and output sequence
        X.append(seq_X)
        y.append(seq_y)
        pass
    X= np.array(X) #Convert X and y into numpy array
    y= np.array(y)
    return X, y

n_steps= 5
X, y= splitSequence(data, n_steps= 5)
print(X)
print(y)
for i in range(len(X)):
    print(X[i], y[i])

n_features= 1
X= X.reshape((X.shape[0], X.shape[1], n_features))
print(X[:2])

# create LSTM model
model= Sequential()
model.add(LSTM(50, activation= 'relu', input_shape= (n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer= Adam(0.01), loss= MeanSquaredError(), metrics= ['accuracy'])
print(model.layers) # view model layers
print(model.summary()) # model summary

history= model.fit(X, y, epochs= 200, verbose= 1) # train the model
loss= history.history['loss']
model.evaluate(X, y, verbose= 1) # evaluate model
pyplot.plot(loss, label= 'Loss') # plot the loss
pyplot.title('Training loss of the model')
pyplot.xlabel('epochs', fontsize= 18)
pyplot.ylabel('loss', fontsize= 18)
pyplot.grid()
pyplot.legend()
pyplot.show()

test_data= np.array([fibonacci(n) for n in range(5)])
test_data= test_data.reshape((1, n_steps, n_features))
print(test_data)
predictNextNumber= model.predict(test_data, verbose= 1) # make predictions of the next number
print(predictNextNumber)

test_data1= np.array([fibonacci(7), fibonacci(8), fibonacci(9), fibonacci(10), fibonacci(11)])
test_data1= test_data1.reshape((1, n_steps, n_features))
print(test_data1)
predictNextNumber1= model.predict(test_data1, verbose= 1) # make prediction with a different part of the sequence
print(predictNextNumber1)
print('Success!')
