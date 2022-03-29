

import numpy as np
import matplotlib.pyplot as plt
import keras

## loading data
from keras.datasets import mnist

## does one-hot encoding for us
from keras.utils import np_utils

## load sequential models
from keras.models import Sequential

## load a dense layer all-to-all connection
from keras.layers import Dense

## loading the MNIST data set
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

## print check shapes
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

## plotting some exa,ples
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i, :, :], cmap=plt.get_cmap("gray"))
plt.show()

## data preparation
## number of pixels
Npixels = X_train.shape[1] * X_train.shape[2]
print("Number of pixels", Npixels)

## flatten arrays to make data faster acessible
X_train = np.reshape(X_train, (X_train.shape[0], Npixels)).astype("float32")
X_test = np.reshape(X_test, (X_test.shape[0], Npixels)).astype("float32")

## then we normalize inputs to have values [0,1]

X_train /= np.max(X_train)
X_test /= np.max(X_test)

## now we have to prepare our labels
## first lets transform them to the one-hot encoding
# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape, Y_test.shape)
Nclasses = Y_test.shape[1]


def createModel():
    ## create model instance
    model = Sequential()
    ## call add member function of model to append some layer
    model.add(Dense(300, input_dim=Npixels, kernel_initializer='normal',
                    activation="relu"))
    ## add first hidden layer always match input and dimension of actual
    ## and previous layer
    model.add(Dense(100, input_dim=300,
                    kernel_initializer="normal", activation="relu"))
    ## add the output layer, since we are doing a classification problem
    ## the number of output layers has to match the number of classes
    ## since we have 10 digits 0..9 ->
    model.add(Dense(Nclasses, input_dim=100,
                    kernel_initializer="normal", activation="softmax"))
    ## and last keras allows us to compile the model to get all computing power
    ## available, and we can define a optimizer and a loss function
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=['accuracy'])
    return model


## create an instance of our model
model = createModel()

## train the model
## the train function allows you to alter number of epochs , batch_size ...,
model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
          epochs=50, batch_size=200, verbose=2)

## lets check how accurate the model is on your test data
# Final evaluation of the model
quality = model.evaluate(X_test, Y_test, verbose=0)
print("Model Error: %.2f%%" % (100 - quality[1] * 100))


model.save("TestModel")

Model_Load = reconstructed_model = keras.models.load_model("TestModel")
num=103
plt.imshow( X_test[ num , : ].reshape( 28 , 28 ) )
## and now predict model outcome
x = np.reshape( X_test[ num , : ] , ( 1 , Npixels ) )
result = Model_Load.predict( x )
print( result.shape )
## extract result from the one-hot encoding by looking
## for where the softmax output is maximal
print( "The digit in your image should be " , np.argmax( Y_test[ num , : ] ) , "and the network says " , np.argmax( result ))
