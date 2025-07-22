import numpy as np
import pandas as pd
import os
import warnings
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# 创建虚拟的 SummaryWriter 类
class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass
    def add_scalar(self, *args, **kwargs):
        pass
    def close(self):
        pass

# 导入其他模块（放在虚拟类之后）
from maddpg import MADDPG
from sim_env import UAVEnv
from buffer import MultiAgentReplayBuffer

warnings.filterwarnings('ignore')

def obs_list_to_state_vector(obs):
    """将观测列表转换为状态向量，处理列表和数组两种类型"""
    # 确保所有观测都是 NumPy 数组
    obs_arrays = []
    for o in obs:
        if isinstance(o, list):
            o = np.array(o)
        obs_arrays.append(o)

    # 展平并拼接
    state = np.hstack([np.ravel(o) for o in obs_arrays])
    return state

def plot_training_curves(score_history, target_score_history, save_dir):
    """绘制训练曲线并保存图像"""
    os.makedirs(save_dir, exist_ok=True)

    # 使用 plt 创建图形
    fig = plt.figure(figsize=(12, 8))

    # 原始分数曲线
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(score_history, label='Chaser Score', alpha=0.6)
    ax1.plot(target_score_history, label='Target Score', alpha=0.6)

    # 移动平均曲线
    window = 100
    if len(score_history) > window:
        score_ma = pd.Series(score_history).rolling(window).mean()
        target_ma = pd.Series(target_score_history).rolling(window).mean()
        ax1.plot(score_ma, label=f'Chaser ({window}-ep MA)', color='blue', linewidth=2)
        ax1.plot(target_ma, label=f'Target ({window}-ep MA)', color='orange', linewidth=2)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Scores')
    ax1.legend()
    ax1.grid(True)

    # 每100回合的平均分数
    ax2 = fig.add_subplot(2, 1, 2)
    avg_scores = []
    avg_target_scores = []
    episodes = []

    for i in range(0, len(score_history), 100):
        if i + 100 <= len(score_history):
            avg_scores.append(np.mean(score_history[i:i+100]))
            avg_target_scores.append(np.mean(target_score_history[i:i+100]))
            episodes.append(i + 100)

    if episodes:  # 确保列表不为空
        ax2.plot(episodes, avg_scores, 'o-', label='Avg Chaser Score (per 100 ep)')
        ax2.plot(episodes, avg_target_scores, 'o-', label='Avg Target Score (per 100 ep)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Average Scores per 100 Episodes')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close(fig)

def print_observation_info(obs):
    """打印观测信息，处理列表和数组两种类型"""
    print("\nObservation details:")
    for i, o in enumerate(obs):
        if isinstance(o, np.ndarray):
            shape = o.shape
            min_val = np.min(o)
            max_val = np.max(o)
            mean_val = np.mean(o)
        elif isinstance(o, list):
            o_arr = np.array(o)
            shape = o_arr.shape
            min_val = np.min(o_arr)
            max_val = np.max(o_arr)
            mean_val = np.mean(o_arr)
        else:
            shape = "unknown"
            min_val = "unknown"
            max_val = "unknown"
            mean_val = "unknown"

        print(f"Agent {i}: shape={shape}, min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")

def format_time(seconds):
    """将秒数格式化为更易读的时间格式"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

if __name__ == '__main__':
    # ====================== 配置参数 ======================
    N_GAMES = 5000       # 总共训练回合数
    MAX_STEPS = 100      # 每回合最大步数限制
    PRINT_INTERVAL = 100 # 多少回合打印一次日志

    evaluate = False     # 训练模式
    best_score = -30     # 初始最佳得分阈值

    # ====================== 初始化环境 ======================
    print("Initializing environment...")
    env = UAVEnv()
    for agent, space in env.observation_space.items():
        print(f"{agent}: {space.shape}")
    n_agents = env.num_agents
    actor_dims = []

    # 打印观测空间信息
    print("\nEnvironment observation space:")
    for agent_id, space in env.observation_space.items():
        print(f"Agent {agent_id}: shape={space.shape}")
        actor_dims.append(space.shape[0])

    # Critic 网络需要所有智能体的观测拼接在一起
    critic_dims = sum(actor_dims)
    n_actions = 2  # 假设所有智能体的动作空间都是二维向量 [ax, ay]

    print(f"\nActor dimensions: {actor_dims}")
    print(f"Critic dimensions: {critic_dims}")
    print(f"Number of agents: {n_agents}")
    print(f"Number of actions: {n_actions}")

    # ====================== 创建目录 ======================
    os.makedirs('tmp/maddpg/', exist_ok=True)  # 模型保存目录
    os.makedirs('training_plots/', exist_ok=True)  # 训练曲线目录

    # ====================== 初始化智能体和缓冲区 ======================
    print("\nInitializing MADDPG agents...")
    maddpg_agents = MADDPG(
        actor_dims, critic_dims, n_agents, n_actions,
        fc1=128, fc2=128,
        alpha=0.00001, beta=0.02,
        scenario='UAV_Round_up',
        chkpt_dir='tmp/maddpg/'
        #alpha=0.0001, beta=0.003，新增gamma=0.7

    )

    memory = MultiAgentReplayBuffer(
        1000000, critic_dims, actor_dims,
        n_actions, n_agents, batch_size=256
    )

    # ====================== 训练准备 ======================
    print('='*50)
    print(f'Starting Training for {N_GAMES} episodes')
    print('='*50)

    total_steps = 0
    score_history = []
    target_score_history = []
    start_time = time.time()
    program_start_time = time.time()  # 记录整个程序开始时间

    # ====================== 创建进度条 ======================
    pbar_episodes = tqdm(total=N_GAMES, desc="Training Episodes", position=0,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    # ====================== 训练循环 ======================
    for episode in range(N_GAMES):
        # 重置环境
        obs = env.reset()
        # 确保观测值是 NumPy 数组
        obs = [np.array(o) if isinstance(o, list) else o for o in obs]

        episode_score = 0
        episode_target_score = 0
        dones = [False] * n_agents
        episode_step = 0

        # 回合内循环
        while not any(dones) and episode_step < MAX_STEPS:
            try:
                # 智能体选择动作
                actions = maddpg_agents.choose_action(obs, total_steps, evaluate)

                # 执行动作
                obs_, rewards, dones = env.step(actions)

                # 确保新观测值是 NumPy 数组
                obs_ = [np.array(o) if isinstance(o, list) else o for o in obs_]

                # 准备Critic网络的输入
                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)

                # 将经验存入回放缓冲区
                memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

                # 训练智能体（每10步一次）
                if total_steps % 10 == 0 and not evaluate:
                    maddpg_agents.learn(memory, total_steps)

                # 更新状态和分数
                obs = obs_
                episode_score += sum(rewards[0:2])  # 追捕方分数
                episode_target_score += rewards[-1]  # 目标方分数
                total_steps += 1
                episode_step += 1

            except Exception as e:
                print(f"\nError at episode {episode}, step {episode_step}:")
                print(e)
                print("Current observation types and shapes:")
                for i, o in enumerate(obs):
                    print(f"Agent {i}: type={type(o)}, shape={o.shape if isinstance(o, np.ndarray) else len(o)}")
                raise

        # 记录分数历史
        score_history.append(episode_score)
        target_score_history.append(episode_target_score)

        # 计算平均分数
        avg_score = np.mean(score_history[-100:]) if len(score_history) >= 100 else np.mean(score_history)
        avg_target_score = np.mean(target_score_history[-100:]) if len(target_score_history) >= 100 else np.mean(target_score_history)

        # 保存最佳模型
        if not evaluate:
            if episode % PRINT_INTERVAL == 0:
                if avg_score > best_score:
                    best_score = avg_score
                    maddpg_agents.save_checkpoint()
                    tqdm.write(f'🔥 Episode {episode}: New best score {avg_score:.1f} > previous best, saving models...')

        # 更新进度条
        elapsed_time = time.time() - program_start_time
        pbar_episodes.update(1)
        pbar_episodes.set_postfix({
            'Score': f'{episode_score:.1f}',
            'Target': f'{episode_target_score:.1f}',
            'Avg Score': f'{avg_score:.1f}',
            'Avg Target': f'{avg_target_score:.1f}',
            'Best': f'{best_score:.1f}',
            'Steps': total_steps,
            'Time': format_time(elapsed_time)
        })

        # 定期打印信息和保存训练曲线
        if episode % PRINT_INTERVAL == 0 and episode > 0:
            elapsed_time = time.time() - start_time
            time_per_episode = elapsed_time / (episode + 1)
            remaining_episodes = N_GAMES - episode - 1
            estimated_time_remaining = remaining_episodes * time_per_episode

            tqdm.write(f'📊 Episode {episode}/{N_GAMES} - '
                       f'Score: {episode_score:.1f}, Target: {episode_target_score:.1f} | '
                       f'Avg: {avg_score:.1f}, Avg Target: {avg_target_score:.1f}')
            tqdm.write(f'⏱️ Elapsed: {format_time(elapsed_time)}, '
                       f'ETA: {format_time(estimated_time_remaining)}')

            # 定期保存训练曲线
            plot_training_curves(score_history, target_score_history, 'training_plots')

    # ====================== 训练结束处理 ======================
    pbar_episodes.close()

    # 保存分数历史
    print('💾 Saving score history...')
    df = pd.DataFrame({
        'Episode': range(len(score_history)),
        'Score': score_history,
        'Target_Score': target_score_history
    })
    df.to_csv('training_scores.csv', index=False)

    # 最终训练曲线
    plot_training_curves(score_history, target_score_history, 'training_plots')

    # 训练完成信息
    total_time = time.time() - program_start_time
    print(f'✅ Training completed in {format_time(total_time)}')
    print(f'   Total episodes: {N_GAMES}')
    print(f'   Total steps: {total_steps}')
    print(f'   Best average score: {best_score:.1f}')