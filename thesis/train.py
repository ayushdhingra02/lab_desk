def train():

    from thesis.ppo import Runner_Args
    from thesis.ppo import Runner
    from thesis.ppo.actor_critic  import AC_Args
    from thesis.ppo.ppo import PPO_Args
    from env import HumanoidEnv
    from gymnasium.wrappers import TimeLimit, OrderEnforcing, PassiveEnvChecker

    from ml_logger import logger

    env = HumanoidEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    # env = OrderEnforcing(env)
    env.reset()

    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(Runner_Args))
    # Cfg = vars(Cfg)


    runner=Runner(env,device="cpu")
    runner.learn(num_learning_iterations=10000, init_at_random_ep_len=False, eval_freq=100)

if __name__ == '__main__':

    from pathlib import Path
    from ml_logger import logger
    from thesis import ROOT_DIR

    stem = Path(__file__).stem
    logger.configure(logger.utcnow(f'sit_to_stand/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{ROOT_DIR}/thesis/runs").resolve(), )
    logger.log_text("""
                    charts:
                    - yKey: train/episode/rew_total/mean
                      xKey: iterations
                    - yKey: train/episode/rew_tracking_lin_vel/mean
                      xKey: iterations
                    - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                      xKey: iterations
                    - yKey: train/episode/rew_action_smoothness_1/mean
                      xKey: iterations
                    - yKey: train/episode/rew_action_smoothness_2/mean
                      xKey: iterations
                    - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                      xKey: iterations
                    - yKey: train/episode/rew_orientation_control/mean
                      xKey: iterations
                    - yKey: train/episode/rew_dof_pos/mean
                      xKey: iterations
                    - yKey: train/episode/command_area_trot/mean
                      xKey: iterations
                    - yKey: train/episode/max_terrain_height/mean
                      xKey: iterations
                    - type: video
                      glob: "videos/*.mp4"
                    - yKey: adaptation_loss/mean
                      xKey: iterations
                    """, filename=".charts.yml", dedent=True)
    train()