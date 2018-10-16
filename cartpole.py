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
episodes_per_update = 20
noise_scaling = 0.1  
parameters = np.random.rand(4) * 2 - 1  
bestreward = 0  
for _ in range(10000):  
    newparams = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling
    reward = 0  
    for _ in range(episodes_per_update):  
        run = run_episode(env,newparams)
        reward += run
    run = run_episode(env,newparams)
    if reward > bestreward:
        bestreward = reward
        parameters = newparams
        if reward == 200:
            break