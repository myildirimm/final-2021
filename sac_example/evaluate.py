import sys
import gym
import torch
import os
import random
import time
import simstar
import numpy as np
import matplotlib.pyplot as plt
from simstarEnv import SimstarEnv
from collections import namedtuple
from collections import defaultdict
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EVAL_EPISODE = 5
NUM_EVAL_STEPS = 4000

def evaluate(port=8080):
    env = SimstarEnv()

    # total length of chosen observation states
    insize = 4 + env.track_sensor_size + env.opponent_sensor_size
    outsize = env.action_space.shape[0]

    hyperparams = {
        "lrvalue": 0.0005,
        "lrpolicy": 0.0001,
        "gamma": 0.97,
        "episodes": 9000,
        "buffersize": 100000,
        "tau": 0.001,
        "batchsize": 64,
        "start_sigma": 0.3,
        "end_sigma": 0,
        "sigma_decay_len": 15000,
        "theta": 0.15,
        "polyak": 0.995,
        "alpha": 0.2,
        "maxlength": 10000,
        "clipgrad": True,
        "hidden": 256,
        "total_explore": 300000.0,
        "epsilon_steady_state": 0.01,
        "epsilon_start": 0.5,
        "autopilot_other_agents": True
    }
    HyperParams = namedtuple("HyperParams", hyperparams.keys())
    hyprm = HyperParams(**hyperparams)

    # Load actor network from checkpoint
    agent = Model(env=env, params=hyprm, insize=insize, outsize=outsize, device=device)

    load_checkpoint(agent)

    total_reward = 0

    for eps in range(NUM_EVAL_EPISODE):
        obs = env.reset()
        state = np.hstack((obs.angle, obs.track, obs.trackPos, obs.speedX, obs.speedY, obs.opponents))
        
        epsisode_reward = 0

        for i in range(NUM_EVAL_STEPS):
            action = agent.get_action(state)
            a_1 = np.clip(action[0], -1, 1)
            a_2 = np.clip(action[1], -1, 1)

            action = np.array([a_1, a_2])

            obs, reward, done, summary = env.step(action)
            next_state = np.hstack((obs.angle, obs.track, obs.trackPos, obs.speedX, obs.speedY, obs.opponents))

            epsisode_reward += reward

            if done:
                # do not restart 
                if "accident" != summary['end_reason']:
                    break
                
            state = next_state

        total_reward += epsisode_reward
        print("Episode: %d, Reward: %.1f"%(i, epsisode_reward))
    
    print("Average reward over %d episodes: %.1f"%(NUM_EVAL_EPISODE,total_reward/NUM_EVAL_EPISODE))

def load_checkpoint(agent): 
    path = "trained_models/master.dat"

    try:
        checkpoint = torch.load(path)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        if 'epsisode_reward' in checkpoint: reward = float(checkpoint['epsisode_reward']) 
    except FileNotFoundError:
        raise FileNotFoundError("model weights are not found")

if __name__ == "__main__":
    evaluate()