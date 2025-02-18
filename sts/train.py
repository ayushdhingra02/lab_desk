import os
from pathlib import Path
from sts.ppo import Runner_Args
from sts.ppo import Runner
from ppo.actor_critic import AC_Args
from sts.ppo.ppo import PPO_Args
from env_prev import HumanoidEnv
from gymnasium.wrappers import TimeLimit, OrderEnforcing, PassiveEnvChecker
from torch.utils.tensorboard import SummaryWriter
import datetime

def train():
    env = HumanoidEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    env = OrderEnforcing(env)
    env.reset()

    # TensorBoard logging setup
    log_dir = f'./runs/{datetime.datetime.now().strftime("%Y-%m-%d/%H%M%S")}'
    writer = SummaryWriter(log_dir)

    # Log hyperparameters
    writer.add_text("Hyperparameters",
                    f"AC_Args: {vars(AC_Args)}\nPPO_Args: {vars(PPO_Args)}\nRunnerArgs: {vars(Runner_Args)}")

    runner = Runner(env, writer=writer,device="cpu")
    runner.learn(num_learning_iterations=10000000, init_at_random_ep_len=False, eval_freq=100 )

    writer.close()  # Close the writer at the end of training

if __name__ == '__main__':
    # TensorBoard logging setup
    from sts import ROOT_DIR
    log_dir = Path(f"{ROOT_DIR}/sts/runs").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define TensorBoard writer location
    stem = Path(__file__).stem
    # writer = SummaryWriter(log_dir=str(log_dir / f'sit_to_stand/{stem}/{datetime.datetime.now().strftime("%H%M%S.%f")}'))

    # Define the charts for TensorBoard (These could also be plotted using TensorBoard's interactive features)
    # writer.add_text("Charts", """
    # - yKey: train/episode/rew_total/mean
    #   xKey: iterations
    # - yKey: train/episode/rew_tracking_lin_vel/mean
    #   xKey: iterations
    # - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
    #   xKey: iterations
    # - yKey: train/episode/rew_action_smoothness_1/mean
    #   xKey: iterations
    # - yKey: train/episode/rew_action_smoothness_2/mean
    #   xKey: iterations
    # - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
    #   xKey: iterations
    # - yKey: train/episode/rew_orientation_control/mean
    #   xKey: iterations
    # - yKey: train/episode/rew_dof_pos/mean
    #   xKey: iterations
    # - yKey: train/episode/command_area_trot/mean
    #   xKey: iterations
    # - yKey: train/episode/max_terrain_height/mean
    #   xKey: iterations
    # - type: video
    #   glob: "videos/*.mp4"
    # - yKey: adaptation_loss/mean
    #   xKey: iterations
    # """, filename="charts.txt")

    train()