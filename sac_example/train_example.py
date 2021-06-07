import numpy as np
import math
import torch
import argparse
from collections import namedtuple
from tensorboardX import SummaryWriter
from simstarEnv import SimstarEnv
import simstar
from model import Model

import wandb



TRACK_NAME = simstar.Environments.CircularRoad
PORT = 8081
HOST = "127.0.0.1"
WITH_OPPONENT = False
SYNC_MODE = True
SPEED_UP = 6

# check if simstar open 
try:
    simstar.Client(host=HOST, port=PORT)
except simstar.TimeoutError or simstar.TransportError :
    raise simstar.TransportError("******* Make sure a Simstar instance is open and running at port %d*******"%(PORT))

wandb.init(project='final-p', entity='ferhatmelih', config={
    "TRACK_NAME": TRACK_NAME,
    "PORT": 8081,
    'WITH_OPPONENT':WITH_OPPONENT,
    'SPEED_UP':SPEED_UP,
})
wandb_config = wandb.config


# training mode: 1      evaluation mode: 0
TRAIN = 1

# ./ trained_model / EVALUATION_NAME_{EVALUATION_REWARD}
EVALUATION_REWARD = 79025

# "best" or "checkpoint"
EVALUATION_NAME = "best"

# ./ model_name / TRAINED_FOLDER
TRAINED_FOLDER = "best_sac"

# if true, then opponent sensor will be used
IS_OPPONENT_SENSOR = True


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SimstarEnv(track=TRACK_NAME,
            add_opponents=WITH_OPPONENT,synronized_mode=SYNC_MODE,speed_up=SPEED_UP,
            port=PORT)
    insize = 4 + env.track_sensor_size
    outsize = env.action_space.shape[0]

    if IS_OPPONENT_SENSOR:
        insize += env.opponent_sensor_size

    hyperparams = {
        "lrvalue": 0.0005,
        "lrpolicy": 0.0001,
        "gamma": 0.97,
        "episodes": 50000,
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

    agent = Model(env, hyprm, insize, outsize, device)
    
    if TRAIN:
        #load_model(model_name=model_name, folder_name=TRAINED_FOLDER, agent=agent, reward=EVALUATION_REWARD, name=EVALUATION_NAME)
        writer = SummaryWriter(comment="_model")
    else:
        load_model(folder_name=TRAINED_FOLDER, agent=agent, reward=EVALUATION_REWARD, name=EVALUATION_NAME)

    best_reward = 0.0
    average_reward = 0.0
    total_average_reward = 0.0
    total_reward = []
    total_steps = 0

    for eps in range(hyprm.episodes):
        obs = env.reset()

        if IS_OPPONENT_SENSOR:
            state = np.hstack((obs.angle, obs.speedX, obs.speedY, obs.opponents, obs.track, obs.trackPos))
        else:
            state = np.hstack((obs.angle, obs.speedX, obs.speedY, obs.track, obs.trackPos))

        episode_reward = 0.0

        for step in range(hyprm.maxlength):
            
            action = np.array(agent.select_action(state=state))
            obs, reward, done, _ = env.step(action)

            if IS_OPPONENT_SENSOR:
                next_state = np.hstack((obs.angle, obs.speedX, obs.speedY, obs.opponents, obs.track, obs.trackPos))
            else:
                next_state = np.hstack((obs.angle, obs.speedX, obs.speedY, obs.track, obs.trackPos))
            
            if (math.isnan(reward)):
                print("\nBad Reward Found\n")
                break
            
            episode_reward += reward

            if TRAIN:
                agent.memory.push(state, action, reward, next_state, done)
                if len(agent.memory.memories) > hyprm.batchsize:
                    agent.update(agent.memory.sample(hyprm.batchsize))

            if done:
                break

            state = next_state

            if np.mod(step, 15) == 0:
                print("Episode:", eps+1, " Step:", step, " Action:", action, " Reward:", reward)

        process = ((eps+1) / hyprm.episodes) * 100

        total_average_reward = average_calculation(total_average_reward, eps+1, episode_reward)
        
        total_reward.append(episode_reward)
        average_reward = torch.mean(torch.tensor(total_reward[-20:])).item()

        total_steps = total_steps + step
        lap_progress = env.progress_on_road

        if TRAIN:
            if (eps+1) % 100 == 0:
                print("Checkpoint is Saved !")
                save_model(agent=agent, reward=episode_reward, name="checkpoint")

            if episode_reward > best_reward:
                print("Model is Saved, Best Reward is Achieved !")
                best_reward = episode_reward
                save_model(agent=agent, reward=best_reward, name="best")
        
        
            tensorboard_writer(writer, eps+1, step, total_average_reward, average_reward, episode_reward, best_reward, total_steps,lap_progress)

        print("\nProcess: {:2.1f}%, Total Steps: {:d},  Episode Reward: {:2.3f},  Best Reward: {:2.2f},  Total Average Reward: {:2.2f}\n".format(process, total_steps, episode_reward, best_reward, total_average_reward), flush=True)
    print("")


def average_calculation(prev_avg, num_episodes, new_val):
    total = prev_avg * (num_episodes - 1)
    total = total + new_val
    return np.float(total / num_episodes)


def tensorboard_writer(writer, eps, step_number, total_average_reward, average_reward, episode_reward, best_reward, total_steps,
    lap_progress):
    writer.add_scalar("step number - episode" , step_number, eps)
    writer.add_scalar("episode reward", episode_reward, eps)
    writer.add_scalar("average reward - episode", average_reward, eps)
    writer.add_scalar("total average reward - episode", total_average_reward, eps)
    writer.add_scalar("average reward - total steps", average_reward, total_steps)
    writer.add_scalar("total average reward - total steps", total_average_reward, total_steps)
    writer.add_scalar("best reward - episode", best_reward, eps)
    writer.add_scalar("best reward - total steps", best_reward, total_steps)

    wandb.log({"step number":step_number,"episode":eps,"episode reward": episode_reward,
        "average reward": average_reward, "total average reward":total_average_reward, 
        "best reward": best_reward, "total_steps":total_steps,
        "lap_progress":lap_progress
        })







def save_model(agent, reward, name):
    path = "saved_models/" + name + "_" + str(int(reward)) + ".dat"

    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic1_state_dict': agent.critic1.state_dict(),
        'critic2_state_dict': agent.critic2.state_dict(),
        'episode_reward': reward
        }, path)


def load_model(folder_name, agent, reward, name):
    try:
        path = "trained_models/" + folder_name + "/" + name + "_" + str(int(reward)) + ".dat"
        checkpoint = torch.load(path)

        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])

    except FileNotFoundError:
        print("Checkpoint Not Found")
        return


if __name__ == "__main__":
    train()
