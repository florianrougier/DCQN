# Snake: Deep Convolutional Q-Learning - Environment file

import numpy as np
import pygame as pg

class Environment():

    def __init__(self, waitTime): #waitime variable is used in order to be able to play the game

        self.width = 880
        self.height = 880
        self.nRows = 10
        self.nColumns = 10
        self.initSnakeLen = 2
        self.defReward = -0.03 #living penalty
        self.negReward = -1. #penalty for loosing
        self.posReward = 2. #reward when eating an apple
        self.waitTime = waitTime


        if self.initSnakeLen > self.nRows / 2:
            self.initSnakeLen = int(self.nRows / 2)

        self.screen = pg.display.set_mode((self.width, self.height))

        self.snakePos = list()
        self.screenMap = np.zeros((self.nRows, self.nColumns))

        for i in range(self.initSnakeLen):
            self.snakePos.append((int(self.nRows / 2) + i, int(self.nColumns / 2)))
            self.screenMap[int(self.nRows / 2) + i][int(self.nColumns / 2)] = 0.5

        self.applePos = self.placeApple()

        self.drawScreen()
        self.collected = False

        self.lastMove = 0

    def placeApple(self):
        posx = np.random.randint(0, self.nColumns)
        posy = np.random.randint(0, self.nRows)
        while self.screenMap[posy][posx] == 0.5:
            posx = np.random.randint(0, self.nColumns)
            posy = np.random.randint(0, self.nRows)

        self.screenMap[posy][posx] = 1

        return (posy, posx)


    def drawScreen(self):

        self.screen.fill((0, 0, 0))

        cellWidth = self.width / self.nColumns
        cellHeight = self.height / self.nRows

        for i in range(self.nRows):
            for j in range(self.nColumns):
                if self.screenMap[i][j] == 0.5:
                    pg.draw.rect(self.screen, (255, 255, 255), (j*cellWidth + 1, i*cellHeight + 1, cellWidth - 2, cellHeight - 2))
                elif self.screenMap[i][j] == 1:
                    pg.draw.circle(self.screen, (255, 0, 0), (int((j+0.5)*cellWidth), int((i+0.5)*cellHeight)), cellWidth // 2)

        pg.display.flip()

    def moveSnake(self, nextPos, col):

        self.snakePos.insert(0, nextPos)

        if not col:
            self.snakePos.pop(len(self.snakePos) - 1)

        self.screenMap = np.zeros((self.nRows, self.nColumns))

        for i in range(len(self.snakePos)):
            self.screenMap[self.snakePos[i][0]][self.snakePos[i][1]] = 0.5

        if col:
            self.applePos = self.placeApple()
            self.collected = True

        self.screenMap[self.applePos[0]][self.applePos[1]] = 1

    def step(self, action):
        # action = 0 -> up
        # action = 1 -> down
        # action = 2 -> right
        # action = 3 -> left
        gameOver = False
        reward = self.defReward
        self.collected = False

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return

        snakeX = self.snakePos[0][1]
        snakeY = self.snakePos[0][0]

        if action == 1 and self.lastMove == 0:
            action = 0
        if action == 0 and self.lastMove == 1:
            action = 1
        if action == 3 and self.lastMove == 2:
            action = 2
        if action == 2 and self.lastMove == 3:
            action = 3

        if action == 0:
            if snakeY > 0:
                if self.screenMap[snakeY - 1][snakeX] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY - 1][snakeX] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY - 1, snakeX), True)
                elif self.screenMap[snakeY - 1][snakeX] == 0:
                    self.moveSnake((snakeY - 1, snakeX), False)
            else:
                gameOver = True
                reward = self.negReward

        elif action == 1:
            if snakeY < self.nRows - 1:
                if self.screenMap[snakeY + 1][snakeX] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY + 1][snakeX] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY + 1, snakeX), True)
                elif self.screenMap[snakeY + 1][snakeX] == 0:
                    self.moveSnake((snakeY + 1, snakeX), False)
            else:
                gameOver = True
                reward = self.negReward

        elif action == 2:
            if snakeX < self.nColumns - 1:
                if self.screenMap[snakeY][snakeX + 1] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY][snakeX + 1] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY, snakeX + 1), True)
                elif self.screenMap[snakeY][snakeX + 1] == 0:
                    self.moveSnake((snakeY, snakeX + 1), False)
            else:
                gameOver = True
                reward = self.negReward

        elif action == 3:
            if snakeX > 0:
                if self.screenMap[snakeY][snakeX - 1] == 0.5:
                    gameOver = True
                    reward = self.negReward
                elif self.screenMap[snakeY][snakeX - 1] == 1:
                    reward = self.posReward
                    self.moveSnake((snakeY, snakeX - 1), True)
                elif self.screenMap[snakeY][snakeX - 1] == 0:
                    self.moveSnake((snakeY, snakeX - 1), False)
            else:
                gameOver = True
                reward = self.negReward

        self.drawScreen()

        self.lastMove = action

        pg.time.wait(self.waitTime)

        return self.screenMap, reward, gameOver


    def reset(self):
        self.screenMap  = np.zeros((self.nRows, self.nColumns))
        self.snakePos = list()

        for i in range(self.initSnakeLen):
            self.snakePos.append((int(self.nRows / 2) + i, int(self.nColumns / 2)))
            self.screenMap[int(self.nRows / 2) + i][int(self.nColumns / 2)] = 0.5

        self.screenMap[self.applePos[0]][self.applePos[1]] = 1

        self.lastMove = 0

        self.drawScreen()

if __name__ == '__main__':
     env = Environment(100)
     gameOver = False
     start = False
     action = 0
     while True:
          for event in pg.event.get():
               if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE and not start:
                         start = True
                    elif event.key == pg.K_SPACE and start:
                         start = False
                    if event.key == pg.K_UP:
                         action = 0
                    elif event.key == pg.K_DOWN:
                         action = 1
                    elif event.key == pg.K_RIGHT:
                         action = 2
                    elif event.key == pg.K_LEFT:
                         action = 3

          if start:
               _, _, gameOver = env.step(action)

          if gameOver:
               start = False
               gameOver = False
               env.reset()
               action = 0
