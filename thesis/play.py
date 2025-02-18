from env import HumanoidEnv
from gymnasium.wrappers import TimeLimit, OrderEnforcing, PassiveEnvChecker
import torch
from tqdm import tqdm
import numpy as np
import glob
import pickle as pkl

def load_policy():
    body = torch.jit.load('D:/Ayush Dhingra/Thesis/thesis/ppo/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load('D:/Ayush Dhingra/Thesis/thesis/ppo/checkpoints/adaptation_module_latest.jit')
    # D:\Ayush
    # Dhingra\Thesis\thesis\runs\sit_to_stand\2025 - 02 - 04\train\193011.427093\checkpoints\adaptation_module_latest.jit

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs.to('cpu'))
        action = body.forward(torch.cat((obs.to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_env(label):
    # dirs = glob.glob(f"../runs/{label}/*")
    # logdir = sorted(dirs)[0]
    #
    # with open(logdir + "/parameters.pkl", 'rb') as file:
    #     pkl_cfg = pkl.load(file)
    #     print(pkl_cfg.keys())
    #     cfg = pkl_cfg["Cfg"]
    #     print(cfg.keys())

        # for key, value in cfg.items():
        #     if hasattr(Cfg, key):
        #         for key2, value2 in cfg[key].items():
        #             setattr(getattr(Cfg, key), key2, value2)

    from env import HumanoidEnv
    from gymnasium.wrappers import TimeLimit, OrderEnforcing, PassiveEnvChecker

    env = HumanoidEnv(render_mode='human')
    env = TimeLimit(env, max_episode_steps=1000)
    # env = OrderEnforcing(env)
    env.reset()

    # load policy
    from ml_logger import logger
    from thesis.ppo.actor_critic import ActorCritic

    policy = load_policy()

    return env, policy

def play():
    from ml_logger import logger

    from pathlib import Path
    from thesis import ROOT_DIR
    import glob
    import os

    label = "gait-conditioned-agility/pretrain-v0/train"

    env, policy = load_env(label)

    num_eval_steps = 2500
    # gaits = {"pronking": [0, 0, 0],
    #          "trotting": [0.5, 0, 0],
    #          "bounding": [0, 0.5, 0],
    #          "pacing": [0, 0, 0.5]}
    #
    # x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    # body_height_cmd = 0.0
    # step_frequency_cmd = 3.0
    # gait = torch.tensor(gaits["trotting"])
    # footswing_height_cmd = 0.08
    # pitch_cmd = 0.0
    # roll_cmd = 0.0
    # stance_width_cmd = 0.25
    #
    # measured_x_vels = np.zeros(num_eval_steps)
    # target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    # joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()
    print (obs[0])
    print(len(obs[0]))
    observ = torch.tensor(obs[0], device='cpu', dtype=torch.float32)
    observ = observ.unsqueeze(0)
    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(observ)
        actions = actions.detach().numpy()
        # env.commands[:, 0] = x_vel_cmd
        # env.commands[:, 1] = y_vel_cmd
        # env.commands[:, 2] = yaw_vel_cmd
        # env.commands[:, 3] = body_height_cmd
        # env.commands[:, 4] = step_frequency_cmd
        # env.commands[:, 5:8] = gait
        # env.commands[:, 8] = 0.5
        # env.commands[:, 9] = footswing_height_cmd
        # env.commands[:, 10] = pitch_cmd
        # env.commands[:, 11] = roll_cmd
        # env.commands[:, 12] = stance_width_cmd
        obs, rew, done,trunc, info = env.step(actions[0])

        # measured_x_vels[i] = env.base_lin_vel[0, 0]
        # joint_positions[i] = env.dof_pos[0, :].cpu()

    # plot target and measured forward velocity
    # from matplotlib import pyplot as plt
    # fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    # axs[0].legend()
    # axs[0].set_title("Forward Linear Velocity")
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Velocity (m/s)")
    #
    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    # axs[1].set_title("Joint Positions")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Joint Position (rad)")
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play()