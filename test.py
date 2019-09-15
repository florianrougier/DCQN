#Snake: Deep Convolutional Q-Learning - Testing file

#Importing the libraries
from environment import Environment
from brain import Brain
import numpy as np

#Defining the parameters
waitTime = 75
nLastStates = 4
filePathToOpen = 'model.h5'

#Initalizing the environment and the Brain
env = Environment(waitTime)
brain = Brain((env.nColumns, env.nRows, nLastStates))
model = brain.loadModel(filePathToOpen)

#Building a function that will reset current state and next state and starting the main loop
def resetStates():

    currentState = np.zeros((1, env.nColumns, env.nRows, nLastStates))

    for i in range(nLastStates):
        currentState[0, :, :, i] = env.screenMap

    return currentState, currentState #Return current state and next state which are the same at the beginning

while True:

    #Resetting the game and staring to play the game
    env.reset()
    currentState, nextState = resetStates()
    gameOver = False
    while not gameOver:

        #Selecting an action to play
        qvalues = model.predict(currentState)[0]
        action = np.argmax(qvalues)

        #Updating the environment and the current state
        frame, _, gameOver = env.step(action)

        #We need to reshape the frame(2D) to add it to the nextState (4D)
        frame = np.reshape(frame, (1, env.nColumns, env.nRows, 1))
        nextState = np.append(nextState, frame, axis = 3)
        nextState = np.delete(nextState, 0, axis = 3)

        currentState = nextState
