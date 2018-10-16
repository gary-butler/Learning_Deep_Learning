import gym
import numpy as np

def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        env.render()
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

env = gym.make('CartPole-v0')
env.reset()
for _ in range(20):
    parameters = np.random.rand(4) * 2 - 1
    run_episode(env, parameters)