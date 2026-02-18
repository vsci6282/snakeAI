import pygame
import time as t
import numpy as np
import stable_baselines3
import random as r
import gymnasium as gym
from gymnasium import spaces
import pdb

willRender = True

class snake(gym.Env):
    pygame.init()
    isRunning = True
    playSpace = np.zeros((12, 12))

    playerDirDict = {1:[0, 1], 2:[-1, 0], 3:[0, -1], 4:[1, 0]}
    playerActionDirDict = {0:-1, 1:0, 2:1}

    actions = 2

    stepsSinceHadFood = 0

    agLeft = 1
    agForward = 2
    agRight = 3

    empty = 0
    snake = 1
    food = 2

    totalFrame = 0
    attempt = 1
    attemptFrame = 0
    def __init__(self, isRendering, hasHumanPlayer, tickSpeed, printsBasicDebug, printsAdvancedDebug, debugFreq):

        self.playerPos = [6, 6]
        self.playerLen = 0
        self.playerTail = [[6, 6] for i in range(0, self.playerLen - 1)]

        self.applePos = [r.randint(0, 11), r.randint(0, 11)]

        self.hasHumanPlayer = hasHumanPlayer
        self.isRendering = isRendering
        self.tickSpeed = tickSpeed
        self.printsBasicDebug = printsBasicDebug
        self.printsAdvancedDebug = printsAdvancedDebug
        self.debugFreq = debugFreq
        self.debugFrame = debugFreq

        self.playerDir = 1
        if isRendering:
            self.screen = pygame.display.set_mode((600, 600))
            self.squareSize = 20
            self.squareMargin = 2
            self.colorDict = {0: (120, 120, 120), 1: (0, 200, 200), 2: (200, 50, 50)}
            self.playSpaceLeftTop = (200, 200)

        self.playSpace[self.playerPos[0], self.playerPos[1]] = 1
        self.playSpace[self.applePos[0], self.applePos[1]] = 2

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
                "grid": self.playSpace.astype(np.uint8)}

    def step(self, action):
        """Evolve environment in response to action and calculate reward"""
        if self.printsAdvancedDebug:
            print(f'applePos: {self.applePos}')
            print(f'playerPos: {self.playerPos}')
            print(f'snakeLen: {self.playerLen}')
            print(f'playertail: {self.playerTail}')
            print(len(self.playerTail))
            print(f'playspace: {self.playSpace}')

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

        if self.printsAdvancedDebug:
            print(f'applePos after playspace set {self.applePos}')

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

        if self.printsAdvancedDebug:
            print(f'action: {action}')

        if self.hasHumanPlayer:
            self.playerDir += self.playerActionDirDict[action]
        else:
            self.playerDir += self.playerActionDirDict[action.item()]
        self.playerDir = (self.playerDir-1)%4+1

        self.playerPos[0] += self.playerDirDict[self.playerDir][0]
        self.playerPos[1] += self.playerDirDict[self.playerDir][1]

        postDistToFood = np.abs(self.playerPos[0]-self.applePos[0])

        if self.printsAdvancedDebug:
            print(f'applepos after postdistofood calc {self.applePos}')

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
            if self.printsBasicDebug:
                self.attempt += 1
                self.attemptFrame = 0
            done = True
            reward -= 20

        self.attemptFrame += 1
        self.totalFrame += 1
        self.debugFrame -= 1
        if self.debugFrame <= 0:
            pdb.set_trace()
            self.debugFrame = self.debugFreq
        if self.printsBasicDebug:
            print(f'total frames: {self.totalFrame}, frames since last attempt: {self.attemptFrame}, attempt # {self.attempt}, frames until next debug: {self.debugFrame}')
        if self.isRendering:
            t.sleep(1/self.tickSpeed)
            self.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        return  self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.playerPos = [6, 6]
        self.playerLen = 0
        self.playerTail = [[6, 6] for i in range(0, self.playerLen - 1)]

        self.applePos = [r.randint(0, 11), r.randint(0, 11)]
        while self.applePos in self.playerTail + self.playerPos:
            self.applePos = [r.randint(0, 11), r.randint(0, 11)]

        self.playerDir = 1
        self.playSpace = np.zeros((12, 12))
        self.playSpace[self.playerPos[0], self.playerPos[1]] = 1
        self.playSpace[self.applePos[0], self.applePos[1]] = 2
        return self._get_obs(), {}

    def render(self):
        for coords, value in np.ndenumerate(self.playSpace):
            rectToBeDrawn = pygame.Rect(self.playSpaceLeftTop[0] + (coords[0] * (self.squareSize + self.squareMargin)),
                                        self.playSpaceLeftTop[1] + (coords[1] * (self.squareSize + self.squareMargin)),
                                        self.squareSize, self.squareSize)
            pygame.draw.rect(self.screen, self.colorDict[value], rectToBeDrawn)

        pygame.display.flip()

    def close(self):
        pass

    def getDistToFood(self):
        return np.abs(self.playerPos[0]-self.applePos[0]) + np.abs(self.playerPos[1]-self.applePos[1])


running = False
game = snake(True, False, 4, True, False, 10)
act = 0
while running and game.hasHumanPlayer:
    act = 2
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                act = 1
            elif event.key == pygame.K_a:
                act = 0
            elif event.key == pygame.K_d:
                act = 2
        if event.type == pygame.QUIT:
            running = False
    game.screen.fill("black")
    info = game.step(act)
    if info[2]:
        running = False
    game.render()
    pygame.display.flip()

    t.sleep(0.3)

print('starting up')
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import results_plotter
from stable_baselines3 import PPO
import os
#Logging
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

env = snake(True, False, 4, True, False, 10)

env = Monitor(env, log_dir)

eval_callback = EvalCallback(env, best_model_save_path='./log/',
                             log_path='./log/', eval_freq=5000,
                             deterministic=False, render=False)

#Train the agent
max_total_step_num = 1e6


def learning_rate_schedule(progress_remaining):
    start_rate = 0.0001 #0.0003
    #Can do more complicated ones like below
    #stepnum = max_total_step_num*(1-progress_remaining)
    #return 0.003 * np.piecewise(stepnum, [stepnum>=0, stepnum>4e4, stepnum>2e5, stepnum>3e5], [1.0,0.5,0.25,0.125 ])
    return start_rate * progress_remaining #linearly decreasing

PPO_model_args = {
    "learning_rate": learning_rate_schedule, #decreasing learning rate #0.0003 #can be set to constant
    "gamma": 0.99, #0.99, discount factor for futurer rewards, between 0 (only immediate reward matters) and 1 (future reward equivalent to immediate),
    "verbose": 0, #change to 1 to get more info on training steps
    #"seed": 137, #fixing the random seed
    "ent_coef": 0.0, #0, entropy coefficient, to encourage exploration
    "clip_range": 0.2 #0.2, very roughly: probability of an action can not change by more than a factor 1+clip_range
}
starttime = t.time()
model = PPO('MultiInputPolicy', env,**PPO_model_args)
#Load previous best model parameters, we start from that
if os.path.exists("log/best_model.zip"):
    model.set_parameters("log/best_model.zip")
model.learn(max_total_step_num,callback=eval_callback)
dt = t.time()-starttime
print("Calculation took %g hr %g min %g s"%(dt//3600, (dt//60)%60, dt%60) )

results_plotter.plot_results(["log"], 1e7, results_plotter.X_TIMESTEPS,'')

#check_env(env, warn=True) #If the environment doesn't follow the interface, an error will be thrown
#print('env checked')

observation, info = env.reset() # Reset environment to start a new episode

print('env reseted')
print(f"Starting observation: {observation}")

'''
episode_over = True
total_reward = 0

while not episode_over:
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()

pygame.quit()
'''
