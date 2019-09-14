#Snake: Deep Convolutional Q-Learning - Testing file

#Importing the libraries
from environment import Environment
from brain import Brain
import numpy as np

#Defining the parameters
waitTime = 75
nLastStates = 4
filepathToOpen = 'model.h5'

#Initializing the Environment and the Brain
env = Environment(waitTime)
brain = Brain((env.nColumns, env.nRows, nLastStates))
model = brain.loadModel(filepathToOpen)

#Building a function that will reset current state and next state and starting the main loop
def resetStates():
     
     currentState = np.zeros((1, env.nColumns, env.nRows, nLastStates))
     
     for i in range(nLastStates):
          currentState[0, :, :, i] = env.screenMap
     
     return currentState, currentState

while True:
     
     #Resetting the game and starting to play the game
     env.reset()
     currentState, nextState = resetStates()
     gameOver = False
     while not gameOver:
          
          #Selecting an action to play
          qvalues = model.predict(currentState)[0]
          action = np.argmax(qvalues)
          
          #Updating the environment and the current state
          frame, _, gameOver = env.step(action)

          frame = np.reshape(frame, (1, env.nColumns, env.nRows, 1))
          nextState = np.append(nextState, frame, axis = 3)
          nextState = np.delete(nextState, 0, axis = 3)
          
          currentState = nextState

