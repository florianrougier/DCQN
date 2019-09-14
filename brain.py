#Snake: Deep Convolutional Q-Learning - Brain file

#Importing the libraries
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

#Building the Brain class
class Brain():
     
     def __init__(self, inputShape, lr = 0.005):
          self.inputShape = inputShape
          self.learningRate = lr
          self.numOutputs = 4
          
          #Creating the neural network
          self.model = Sequential()
          
          self.model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = self.inputShape))
          
          self.model.add(MaxPooling2D((2, 2)))
          
          self.model.add(Conv2D(64, (2,2), activation = 'relu'))
          
          self.model.add(Flatten())
          
          self.model.add(Dense(256, activation = 'relu'))
          
          self.model.add(Dense(self.numOutputs))
          
          self.model.compile(optimizer = Adam(lr = self.learningRate), loss = 'mean_squared_error')
          
     #Building a method that will load a model
     def loadModel(self, filepath):
          self.model = load_model(filepath)
          return self.model

