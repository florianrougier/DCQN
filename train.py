#Snake: Deep Convolutional Q-Learning - Training file

#Importing the libraries
from environment import Environment
from brain import Brain
from dqn import Dqn
import numpy as np
import matplotlib.pyplot as plt

#Defining the parameters
learningRate = 0.0001
maxMemory = 60000
gamma = 0.9
batchSize = 32
nLastStates = 4

epsilon = 1.
epsilonDecayRate = 0.0002
minLastEpsilon = 0.05

filePathToSave = 'model2.h5'

#Initializing the environment, the brain and the Experience Replay Memory
env = Environment(0)
brain = Brain((env.nColumns, env.nRows, nLastStates), learningRate)
model = brain.model
DQN = Dqn(maxMemory, gamma)

#Building a function that will reset current state and next state
def resetStates():
    currentState = np.zeros((1, env.nColumns, env.nRows, nLastStates))

    for i in range(nLastStates):
        currentState[0, :, :, i] = env.screenMap

    return currentState, currentState #Return current state and next state which are the same at the beginning

#Starting the main loop
epoch = 0
nCollected = 0
maxNCollected = 0
totNCollected = 0
scores = list()

while True:
    epoch += 1

    #Reseting the environment and starting to play the game
    env.reset()
    currentState, nextState = resetStates()
    gameOver = False
    while not gameOver:

        #Selecting an action to play
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, 4)
        else:
            qvalues = model.predict(currentState)[0]
            action = np.argmax(qvalues)

        #Updating the Environment
        frame, reward, gameOver = env.step(action)

        #We need to reshape the frame(2D) to add it to the nextState (4D)
        frame = np.reshape(frame, (1, env.nColumns, env.nRows, 1))
        nextState = np.append(nextState, frame, axis = 3)
        nextState = np.delete(nextState, 0, axis = 3)

        #Remembering new experience and training the AI
        DQN.remember([currentState, action, reward, nextState], gameOver)
        inputs, targets = DQN.getBatch(model, batchSize)
        model.train_on_batch(inputs, targets)

        #Updating the score and current state
        if env.collected:
            nCollected += 1

        currentState = nextState

    #Updating the epsilon and saving the model
    epsilon -= epsilonDecayRate
    epsilon = max(epsilon, minLastEpsilon)

    if nCollected > maxNCollected and nCollected > 2:
        model.save(filePathToSave)
        maxNCollected = nCollected

    #Displaying the results
    totNCollected += nCollected
    nCollected = 0

    if epoch % 100 == 0 and epoch != 0:
        scores.append(totNCollected / 100)
        totNCollected = 0
        plt.plot(scores)
        plt.xlabel('Epoch / 100')
        plt.ylabel('Average collected')
        plt.show()

    print('Epoch ' + str(epoch) + ' Current Best: ' + str(maxNCollected) + ' Epsilon:  {:.5f}'.format(epsilon))
