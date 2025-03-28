import time
from collections import deque
import copy
import os

import torch
from ml_logger import logger
from params_proto import PrefixProto

from thesis.ppo.actor_critic import ActorCritic
from thesis.ppo.rollout_storage import RolloutStorage

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

class DataCaches:
    def __init__(self, curriculum_bins):
        from thesis.ppo.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)

class Runner_Args(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 100  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 100# check for potential saves every this many iterations
    save_video_interval = False
    log_freq = 5

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True

class Runner:

    def __init__(self,env,device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env

        actor_critic = ActorCritic(self.env.unwrapped.num_obs,
                                   self.env.unwrapped.num_privileged_obs,
                                   self.env.unwrapped.num_obs_history,
                                   self.env.unwrapped.num_actions,
                                   ).to(self.device)

        if Runner_Args.resume:
            # load pretrained weights from resume_path
            from ml_logger import ML_Logger
            loader = ML_Logger(root="http://escher.csail.mit.edu:8080",
                               prefix=Runner_Args.resume_path)
            weights = loader.load_torch("checkpoints/ac_weights_last.pt")
            actor_critic.load_state_dict(state_dict=weights)

            if hasattr(self.env, "curricula") and Runner_Args.resume_curriculum:
                # load curriculum state
                distributions = loader.load_pkl("curriculum/distribution.pkl")
                distribution_last = distributions[-1]["distribution"]
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = Runner_Args.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.unwrapped.num_train_envs, self.num_steps_per_env, [self.env.unwrapped.num_obs],
                              [self.env.unwrapped.num_privileged_obs], [self.env.unwrapped.num_obs_history], [self.env.unwrapped.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500,
              eval_expert=False):

        from ml_logger import logger

        logger.start('start', 'epoch', 'episode', 'run', 'step')

        if init_at_random_ep_len:
            self.env.unwrapped.episode_length_buf = torch.randint_like(self.env.unwrapped.episode_length_buf,
                                                             high=int(self.env.unwrapped.max_episode_length))

            # split train and test envs
        num_train_envs = self.env.unwrapped.num_train_envs

        obs_dict = self.env.unwrapped.get_observations()
        obs = torch.tensor(obs_dict, device=self.device, dtype=torch.float32)
        obs=obs.unsqueeze(0)
        privileged_obs = obs  # Or any appropriate value
        obs_history = obs  # Or any appropriate value
        # obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        # obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
        #     self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.unwrapped.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.unwrapped.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs] )
                    # print(obs_history.shape)
                    if eval_expert:
                        actions_eval = self.alg.actor_critic.act_teacher(obs_history[num_train_envs:],
                                                                         privileged_obs[num_train_envs:])
                    else:
                        actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:]) #num_train_envs:
                    actions_train = actions_train.detach().numpy()
                    ret = self.env.step(actions_train[0])
                    obs_dict, rewards, dones,tuncation, infos = ret

                    # obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                    #     "obs_history"]
                    obs=obs_dict
                    obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
                    obs=obs.unsqueeze(0)
                    privileged_obs=obs
                    obs_history=obs
                    # obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                    #     self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)

                    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                    dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
                    if rewards.dim() == 0:
                        rewards = rewards.unsqueeze(0)
                    else:
                        rewards=rewards[:num_train_envs]
                    if dones.dim() == 0:
                        dones = dones.unsqueeze(0)
                    else:
                        dones=dones[:num_train_envs]
                    self.alg.process_env_step(rewards,dones , infos)

                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])

                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])

                    if 'curriculum' in infos:
                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])
                # self.alg.compute_returns(obs_history, privileged_obs)

                if it % curriculum_dump_freq == 0:
                    logger.save_pkl({"iteration": it,
                                     **caches.slot_cache.get_summary(),
                                     **caches.dist_cache.get_summary()},
                                    path=f"curriculum/info.pkl", append=True)

                    if 'curriculum/distribution' in infos:
                        logger.save_pkl({"iteration": it,
                                         "distribution": distribution},
                                        path=f"curriculum/distribution.pkl", append=True)
            self.env.reset()
            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            logger.store_metrics(
                # total_time=learn_time - collection_time,
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                mean_decoder_loss=mean_decoder_loss,
                mean_decoder_loss_student=mean_decoder_loss_student,
                mean_decoder_test_loss=mean_decoder_test_loss,
                mean_decoder_test_loss_student=mean_decoder_test_loss_student,
                mean_adaptation_module_test_loss=mean_adaptation_module_test_loss
            )

            if Runner_Args.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.unwrapped.num_envs
            if logger.every(Runner_Args.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:
                logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
                logger.job_running()

            if it % Runner_Args.save_interval == 0:
                with logger.Sync():
                    checkpoint_dir = 'D:/Ayush Dhingra/Thesis/checkpoints1/'
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(self.alg.actor_critic.state_dict(), f"{checkpoint_dir}/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"{checkpoint_dir}/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = './tmp/legged_data'

                    os.makedirs(path, exist_ok=True)

                    adaptation_module_path = f'{path}/adaptation_module_latest.jit'
                    adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                    traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    traced_script_adaptation_module.save(adaptation_module_path)

                    body_path = f'{path}/body_latest.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(body_path)

                    logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/")
                    logger.upload_file(file_path=body_path, target_path=f"checkpoints/")

            self.current_learning_iteration += num_learning_iterations

        with logger.Sync():
            checkpoint_dir = 'D:/Ayush Dhingra/Thesis/checkpoints1/'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(self.alg.actor_critic.state_dict(), f"{checkpoint_dir}/ac_weights_{it:06d}.pt")
            logger.duplicate(f"{checkpoint_dir}/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            path = './tmp/legged_data'

            os.makedirs(path, exist_ok=True)

            adaptation_module_path = f'{path}/adaptation_module_latest.jit'
            adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            traced_script_adaptation_module.save(adaptation_module_path)

            body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

            logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/" )
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/")

    def log_video(self, it):
        if it - self.last_recording_it >= Runner_Args.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            # logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                # logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)




    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert