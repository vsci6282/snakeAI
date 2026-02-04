import pygame
import time as t
import numpy as np
import stable_baselines3
import gym
from gym import spaces

willRender = True

class snake(gym.Env):
    pygame.init()
    isRunning = True
    playSpace = np.zeros((12, 12))
    playerPos = [6, 6]
    playerLen = 3
    playerTail = []
    playerDir = 1
    actions = 3

    agLeft = 1
    agForward = 2
    agRight = 3

    empty = 0
    snake = 1
    food = 2
    def __init__(self, grid_size=12):
        if willRender:
            self.screen = pygame.display.set_mode((600, 600))
            self.squareSize = 20
            self.squareMargin = 2
            self.colorDict = {0: (120, 120, 120), 1: (0, 200, 200), 2: (200, 50, 50)}
            self.playSpaceLeftTop = (200, 200)
        #self.action_space = ... #define possible actions
        #self.observation_space = ... #format of observations

    def _get_obs(self):
        """Calculates the observations (input) from current state"""
        ...
        #return observation #in format of self.observation_space
    def reset(self):
        """Resetting the environment (i.e., starting a new game)"""
        ...
        return self._get_obs()

    def step(self, action):
        """Evolve environment in response to action and calculate reward"""
        self.playSpace[self.playerPos[0], self.playerPos[1]] = 0
        if len(self.playerTail) > 0:
            self.playerTail.pop(0)
        if len(self.playerTail) <= self.playerLen:
            self.playerTail.append([self.playerPos[0], self.playerPos[1]])
        if action == 1:
            self.playerDir -= 1
        elif action == 2:
            pass
        elif action == 3:
            self.playerDir += 1
        self.playerDir = (self.playerDir-1)%4+1
        if self.playerDir == 1:
            self.playerPos[1] += 1
        if self.playerDir == 2:
            self.playerPos[0] -= 1
        if self.playerDir == 3:
            self.playerPos[1] -= 1
        if self.playerDir == 4:
            self.playerPos[0] += 1
        try:
            self.playSpace[self.playerPos[0], self.playerPos[1]] = 1
            for pos in self.playerTail:
                pass
                #self.playSpace[pos[0], pos[1]] = 1
        except IndexError:
            pygame.quit()
        #return  self._get_obs(), reward, done, info

    def render(self):
        for coords, value in np.ndenumerate(self.playSpace):
            rectToBeDrawn = pygame.Rect(self.playSpaceLeftTop[0] + (coords[0] * (self.squareSize + self.squareMargin)),
                                        self.playSpaceLeftTop[1] + (coords[1] * (self.squareSize + self.squareMargin)),
                                        self.squareSize, self.squareSize)
            pygame.draw.rect(self.screen, self.colorDict[value], rectToBeDrawn)
'''
def render():
    for coords, value in np.ndenumerate(playSpace):
        rectToBeDrawn = pygame.Rect(playSpaceLeftTop[0]+(coords[0]*(squareSize+squareMargin)),
                                    playSpaceLeftTop[1]+(coords[1]*(squareSize+squareMargin)),
                                    squareSize, squareSize)
        pygame.draw.rect(screen, colorDict[value], rectToBeDrawn)

def updatePlaySpace():
    playSpace[playerPos[0], playerPos[1]] = 0
    if len(playerTail) > 0:
        playerTail.pop(0)
    if len(playerTail) <= playerLen:
        playerTail.append([playerPos[0], playerPos[1]])
    print(playerTail)
    if playerDir == 1:
        playerPos[1] += 1
    if playerDir == 2:
        playerPos[0] -= 1
    if playerDir == 3:
        playerPos[1] -= 1
    if playerDir == 4:
        playerPos[0] += 1
    try:
        playSpace[playerPos[0], playerPos[1]] = 1
        for pos in playerTail:
            playSpace[pos[0], pos[1]] = 1
    except IndexError:
        pygame.quit()


'''
running = True
game = snake()
act = 0
while running:
    act = 2
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                act = 2
            elif event.key == pygame.K_a:
                act = 1
            elif event.key == pygame.K_d:
                act = 3
        if event.type == pygame.QUIT:
            running = False
    print(act)
    game.screen.fill("black")
    game.step(act)
    game.render()
    pygame.display.flip()

    t.sleep(0.5)

pygame.quit()