import time
from collections import deque
import copy
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from params_proto import PrefixProto

from sts.ppo.actor_critic import ActorCritic
from sts.ppo.rollout_storage import RolloutStorage


class DataCaches:
    def __init__(self, curriculum_bins):
        from thesis.ppo.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class Runner_Args(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 128  # per iteration
    max_iterations = 15000000  # number of policy updates

    # logging
    save_interval = 1000  # check for potential saves every this many iterations
    save_video_interval = False
    log_freq = 5

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True


class Runner:

    def __init__(self, env,writer, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env
        self.writer = writer

        actor_critic = ActorCritic(self.env.unwrapped.num_obs,
                                   self.env.unwrapped.num_privileged_obs,
                                   self.env.unwrapped.num_obs_history,
                                   self.env.unwrapped.num_actions).to(self.device)

        # if Runner_Args.resume:
        #     self.writer.add_text('Resume Info', f"Resuming training from: {Runner_Args.resume_path}", 0)
        #     print(f"Resuming training from: {Runner_Args.resume_path}")
        #
        #     weights = torch.load(os.path.join(Runner_Args.resume_path, "checkpoints/ac_weights_last.pt"),
        #                          map_location=self.device)
        #     actor_critic.load_state_dict(state_dict=weights)
        #     self.writer.add_text('Resume Info', "Loaded pretrained weights from checkpoints/ac_weights_last.pt", 0)
        #     print("Loaded pretrained weights from checkpoints/ac_weights_last.pt")
        #
        #     if hasattr(self.env, "curricula") and Runner_Args.resume_curriculum:
        #         curriculum_path = os.path.join(Runner_Args.resume_path, "curriculum/distribution.pkl")
        #         distributions = torch.load(curriculum_path)
        #         distribution_last = distributions[-1]["distribution"]
        #
        #         for gait_id, gait_name in enumerate(self.env.unwrapped.category_names):
        #             if gait_name:
        #                 self.env.unwrapped.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
        #                 self.writer.add_text('Curriculum Info', f"Gait {gait_name} weights loaded", gait_id)
        #                 print(f"Gait {gait_name} weights loaded")

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = Runner_Args.num_steps_per_env
        self.alg.init_storage(self.env.unwrapped.num_train_envs, self.num_steps_per_env, [self.env.unwrapped.num_obs],
                              [self.env.unwrapped.num_privileged_obs], [self.env.unwrapped.num_obs_history], [self.env.unwrapped.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0
        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500,
              eval_expert=False):

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.unwrapped.episode_length_buf,
                                                             high=int(self.env.unwrapped.max_episode_length))

        num_train_envs = self.env.unwrapped.num_train_envs
        obs_dict = self.env.unwrapped.get_observations()
        # obs_dict = self.env.get_observations()
        obs = obs_dict["obs"]
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        privileged_obs = obs_dict["privileged_obs"]
        privileged_obs = torch.tensor(privileged_obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        obs_history = obs_dict["obs_history"]
        obs_history = torch.tensor(obs_history, device=self.device, dtype=torch.float32).unsqueeze(0)

        self.alg.actor_critic.train()

        rewbuffer = deque(maxlen=100)  # This will store the rewards for averaging
        lenbuffer = deque(maxlen=100)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            steps=0
            episode_reward = 0  # Initialize the reward for this episode
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    steps+=1
                    flag=0
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    actions_train = actions_train.detach().numpy()
                    ret = self.env.step(actions_train[0])
                    obs_dict, rewards, done, truncation, infos = ret

                    obs = obs_dict["obs"]
                    obs = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                    privileged_obs = obs_dict["privileged_obs"]
                    privileged_obs = torch.tensor(privileged_obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                    obs_history = obs_dict["obs_history"]
                    obs_history = torch.tensor(obs_history, device=self.device, dtype=torch.float32).unsqueeze(0)

                    if done or truncation:
                        rewards-=10
                        flag=1
                        # break
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                    dones = torch.tensor(done, dtype=torch.bool).to(self.device)

                    if rewards.dim() == 0:
                        rewards = rewards.unsqueeze(0)
                    if dones.dim() == 0:
                        dones = dones.unsqueeze(0)


                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    # Add the reward to the episode total
                    episode_reward += rewards.mean().item()



                    # Log per step if necessary
                    if 'curriculum' in infos:
                        self.writer.add_scalar('Rewards', rewards.mean().item(), it)
                        self.writer.add_scalar('Episode Length', lenbuffer.mean() if len(lenbuffer) > 0 else 0, it)

                    if flag==1:
                        break

            stop = time.time()
            self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])
            self.env.reset()

            # Log the average reward for this episode
            mean_episode_reward = episode_reward / steps  # Average per episode
            rewbuffer.append(mean_episode_reward)
            self.writer.add_scalar('Episode/Mean Episode Reward', mean_episode_reward, global_step=it)
            self.writer.add_scalar('Episode/Number of steps', steps, global_step=it)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg.update()
            self.writer.add_scalar('Loss/Value Loss', mean_value_loss, global_step=it)
            self.writer.add_scalar('Loss/Surrogate Loss', mean_surrogate_loss, global_step=it)
            self.writer.add_scalar('Loss/Adaptation Module Loss', mean_adaptation_module_loss, global_step=it)
            self.writer.add_scalar('Timing/Iteration Time', stop - start, global_step=it)
            self.writer.flush()

            # Save model and log losses
            if it % Runner_Args.save_interval == 0:
                checkpoint_dir = './checkpoints1_prev1'
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'iteration': it,
                    'model_state_dict': self.alg.actor_critic.state_dict(),
                    'optimizer_state_dict': self.alg.optimizer.state_dict()
                }, f"{checkpoint_dir}/ac_checkpoint_{it:06d}.pt")

        self.writer.close()

    def log_video(self, it):
        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            video_path = f"videos/{it:05d}.mp4"
            print(f"LOGGING VIDEO: {video_path}")
            self.writer.add_video('Training Video', torch.tensor(frames).permute(0, 3, 1, 2), it, fps=1 / self.env.unwrapped.dt)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert