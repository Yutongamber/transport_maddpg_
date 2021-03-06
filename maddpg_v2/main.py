import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from env.transport import Transport
from collections import deque


USE_CUDA = False  # torch.cuda.is_available()

# def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
#     def get_env_fn(rank):
#         def init_env():
#             env = make_env(env_id, discrete_action=discrete_action)
#             env.seed(seed + rank * 1000)
#             np.random.seed(seed + rank * 1000)
#             return env
#         return init_env
#     if n_rollout_threads == 1:
#         return DummyVecEnv([get_env_fn(0)])
#     else:
#         return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def make_parallel_env_transport(env_id, conf, n_rollout_threads, seed, discrete_action =True):
    def get_env_fn(rank):
        def init_env():
            # env = make_env(env_id, discrete_action=discrete_action)
            # env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return Transport(conf)
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
        # return get_env_fn(0)
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    start_time = time.time()
    scores_window = deque(maxlen=100)

    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # transport configuration
    name = 'Materials Transport'
    conf = {
        'n_player': 2,#玩家数量
        'board_width': 11,#地图宽
        'board_height': 11,#地图高
        'n_cell_type': 5,#格子的种类
        'materials': 4,#集散点数量
        'cars': 2,#汽车数
        'planes': 0,#飞机数量
        'barriers': 12,#固定障碍物数量
        'max_step' :100,#最大步数
        'game_name':name,#游戏名字
        'K': 5,#每个K局更新集散点物资数目
        'map_path':'env/map.txt',#存放初始地图
        'cell_range': 6,  # 单格中各维度取值范围（tuple类型，只有一个int自动转为tuple）##?
        'ob_board_width': None,  # 不同智能体观察到的网格宽度（tuple类型），None表示与实际网格相同##?
        'ob_board_height': None,  # 不同智能体观察到的网格高度（tuple类型），None表示与实际网格相同##?
        'ob_cell_range': None,  # 不同智能体观察到的单格中各维度取值范围（二维tuple类型），None表示与实际网格相同##?
    }

    env = make_parallel_env_transport(config.env_id, conf, config.n_rollout_threads, config.seed,
                            config.discrete_action)

    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        score = 0
        # print("Episodes %i-%i of %i" % (ep_i + 1,
        #                                 ep_i + 1 + config.n_rollout_threads,
        #                                 config.n_episodes))

        obs = env.reset() # TODO: TO CHECK
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):

        # while env.is_terminal() is not True:

            env.render()
            # rearrange observations to be per agent, and convert to torch Variable
            # print('step', et_i)
            # print(maddpg.nagents)
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), # 沿着竖直方向将矩阵堆叠起来。
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]

            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            ############################################
            # add
            # actions = actions.astype(int)
            ############################################
            # add: 前两个action
            joint_action = []
            for i in range(2):
                player = []
                for j in range(1):
                    each = [0] * 11
                    idx = np.random.randint(11)
                    each[idx] = 1
                    player.append(each)
                joint_action.append(player)
            for m in range(2):
                joint_action.append([actions[0][m].astype(int).tolist()])

            next_obs, rewards, dones, infos = env.step(joint_action)

            #################################
            agents_action = actions[0]
            #################################

            replay_buffer.push(obs, agents_action, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')

                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)

                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets() #TODO
                maddpg.prep_rollouts(device='cpu')

            score += rewards[0][0]

        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

        scores_window.append(score)
        reward_epi = np.mean(scores_window)
        reward_epi_var = np.var(scores_window)
        logger.add_scalar('results/completion_window' % reward_epi, ep_i)
        logger.add_scalar('results/completion_window' % reward_epi_var, ep_i)
        print('\r Episode {}\t Average Reward: {:.3f}\t Var Reward: {:.3f} \t '.format(ep_i, reward_epi, reward_epi_var) )


    print('=============training time: ', time.time() - start_time)
    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="transport")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents", default="./model")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int) # walker
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--steps_per_update", default=50, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.005, type=float)   # 0.001
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true', default= True)

    config = parser.parse_args()

    run(config)
