import pygame
import time as t
import numpy as np
import stable_baselines3
import random as r
import gym
from gym import spaces

willRender = True

class snake(gym.Env):
    pygame.init()
    isRunning = True
    playSpace = np.zeros((12, 12))

    playerDirDict = {1:[0, 1], 2:[-1, 0], 3:[0, -1], 4:[1, 0]}
    playerActionDirDict = {1:-1, 2:0, 3:1}

    actions = 3

    stepsSinceHadFood = 0

    agLeft = 1
    agForward = 2
    agRight = 3

    empty = 0
    snake = 1
    food = 2
    def __init__(self, hasHumanPlayer):

        self.playerPos = [6, 6]
        self.playerLen = 0
        self.playerTail = [[6, 6] for i in range(0, self.playerLen - 1)]

        self.applePos = [r.randint(0, 11), r.randint(0, 11)]

        self.hasHumanPlayer = hasHumanPlayer

        self.playerDir = 1
        if willRender:
            self.screen = pygame.display.set_mode((600, 600))
            self.squareSize = 20
            self.squareMargin = 2
            self.colorDict = {0: (120, 120, 120), 1: (0, 200, 200), 2: (200, 50, 50)}
            self.playSpaceLeftTop = (200, 200)

        self.action_space = spaces.Discrete(3)
        # The observation space, "position" is the coordinates of the head; "direction" is which way the sanke is heading, "grid" contains the full grid info
        self.observation_space = gym.spaces.Dict(
            spaces={
                "position": gym.spaces.Box(low=0, high=(11), shape=(2,), dtype=np.int32),
                "direction": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int32),
                "grid": gym.spaces.Box(low=0, high=2, shape=(12, 12), dtype=np.uint8),
            })
        #self.action_space = ... #define possible actions
        #self.observation_space = ... #format of observations

    def _get_obs(self):
        """Calculates the observations (input) from current state"""
        direction = np.array(self.playerDirDict[self.playerDir])
        # return observation in the format of self.observation_space
        return {"position": np.array(self.playerPos, dtype=np.int32),
                "direction": direction.astype(np.int32),
                "grid": self.playSpace}

    def reset(self):
        self.playerPos = [6, 6]
        self.playerLen = 0
        self.playerTail = [[6, 6] for i in range(0, self.playerLen - 1)]

        self.applePos = [r.randint(0, 11), r.randint(0, 11)]
        while self.applePos in self.playerTail + self.playerPos:
            self.applePos = [r.randint(0, 11), r.randint(0, 11)]

        self.playerDir = 1
        playSpace = np.zeros((12, 12))
        return self._get_obs()

    def step(self, action):
        """Evolve environment in response to action and calculate reward"""

        print(f'applePos: {self.applePos}')
        print(f'playerPos: {self.playerPos}')
        print(f'snakeLen: {self.playerLen}')
        print(f'playertail: {self.playerTail}')
        print(len(self.playerTail))

        done = False
        reward = 0
        preDistToFood = self.getDistToFood()

        for pos in self.playerTail:
            if pos == self.playerPos:
                done = True
                reward -= 20

        if self.stepsSinceHadFood >= 200:
            done = True

        if self.playerLen >= 143:
            done = True

        self.playSpace[self.applePos[0], self.applePos[1]] = 2

        if len(self.playerTail) < 0:
            self.playSpace[self.playerPos[0], self.playerPos[1]] = 0
        elif len(self.playerTail) < self.playerLen:
            self.playerTail.append([self.playerPos[0], self.playerPos[1]])
        elif len(self.playerTail) == self.playerLen:
            self.playerTail.append([self.playerPos[0], self.playerPos[1]])
            toBeRemoved = self.playerTail[0].copy()
            self.playerTail.pop(0)
            self.playSpace[toBeRemoved[0], toBeRemoved[1]] = 0
        else:
            toBeRemoved = self.playerTail[0].copy()
            self.playerTail.pop(0)
            self.playSpace[toBeRemoved[0], toBeRemoved[1]] = 0

        self.playerDir += self.playerActionDirDict[action]
        self.playerDir = (self.playerDir-1)%4+1

        self.playerPos[0] += self.playerDirDict[self.playerDir][0]
        self.playerPos[1] += self.playerDirDict[self.playerDir][1]

        postDistToFood = np.abs(self.playerPos[0]-self.applePos[0])

        if preDistToFood >= postDistToFood:
            reward += 1
        else:
            reward -= 1

        if self.playerPos == self.applePos:
            reward += 50
            self.stepsSinceHadFood = 0
            print("Father, I have felt the apple perish and rot within my gaping maw. Still, my hunger persists. I desire more.")
            self.playerLen += 1
            self.applePos = [r.randint(0, 11), r.randint(0, 11)]
            while self.applePos in self.playerTail + self.playerPos:
                self.applePos = [r.randint(0, 11), r.randint(0, 11)]
        else:
            self.stepsSinceHadFood += 1

        if 0 <= self.playerPos[0] <= 11 and 0 <= self.playerPos[1] <= 11:
            for pos in self.playerTail+[self.playerPos]:
                self.playSpace[pos[0], pos[1]] = 1
        else:
            print('goodbye, cruel world')
            done = True
            reward -= 20

        return  self._get_obs(), reward, done, {}

    def render(self):
        for coords, value in np.ndenumerate(self.playSpace):
            rectToBeDrawn = pygame.Rect(self.playSpaceLeftTop[0] + (coords[0] * (self.squareSize + self.squareMargin)),
                                        self.playSpaceLeftTop[1] + (coords[1] * (self.squareSize + self.squareMargin)),
                                        self.squareSize, self.squareSize)
            pygame.draw.rect(self.screen, self.colorDict[value], rectToBeDrawn)

    def getDistToFood(self):
        return np.abs(self.playerPos[0]-self.applePos[0]) + np.abs(self.playerPos[1]-self.applePos[1])
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
game = snake(True)
act = 0
while running and game.hasHumanPlayer:
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
    game.screen.fill("black")
    info = game.step(act)
    if info[2]:
        running = False
    game.render()
    pygame.display.flip()

    t.sleep(0.3)

#pygame.quit()
