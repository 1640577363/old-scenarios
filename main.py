import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from tqdm import tqdm
from datetime import datetime # 导入 datetime 模块

# 导入你的自定义模块
from maddpg import MADDPG
from sim_env import UAVEnv
from buffer import MultiAgentReplayBuffer

# --- Matplotlib 字体设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

def obs_list_to_state_vector(obs):
    """将观测列表展平为单一状态向量。"""
    state = np.hstack([np.ravel(o) for o in obs])
    return state

def save_image(env_render, filename):
    """保存环境渲染图像。"""
    image = Image.fromarray(env_render, 'RGBA')
    image = image.convert('RGB')
    image.save(filename)

def format_time(seconds):
    """将秒数格式化为更易读的 Hh Mm Ss 字符串。"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def plot_curves(score_history, target_score_history, save_dir, filename="curves.png", title_prefix=""):
    """绘制并保存得分曲线，可用于训练和评估。"""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(score_history, label='追捕方得分', alpha=0.8)
    ax.plot(target_score_history, label='目标方得分', alpha=0.8)
    ax.set_xlabel('回合数')
    ax.set_ylabel('得分')
    ax.set_title(f'{title_prefix}每回合得分变化')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close(fig)

def plot_losses(actor_losses, critic_losses, save_dir, filename="losses.png"):
    """绘制并保存训练损失曲线。"""
    os.makedirs(save_dir, exist_ok=True)
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    plot_actor_losses = [l for l in actor_losses if l is not None]
    plot_critic_losses = [l for l in critic_losses if l is not None]
    plot_episodes = [idx for idx, l in enumerate(actor_losses) if l is not None]

    if plot_actor_losses:
        ax_loss.plot(plot_episodes, plot_actor_losses, label='Actor Loss', alpha=0.8)
    if plot_critic_losses:
        ax_loss.plot(plot_episodes, plot_critic_losses, label='Critic Loss', alpha=0.8)

    ax_loss.set_xlabel('回合数')
    ax_loss.set_ylabel('损失值')
    ax_loss.set_title('训练损失变化')
    ax_loss.legend()
    ax_loss.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close(fig_loss)

def save_hyperparameters(save_dir, params, filename="hyperparameters.txt"):
    """
    将超参数保存到文本文件中。
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        f.write("--- 训练超参数 ---\n")
        # 直接写入RUN_ID作为第一行，确保其顺序
        if "RUN_ID" in params:
            f.write(f"RUN_ID: {params['RUN_ID']}\n")
            # 移除已写入的RUN_ID，避免重复
            params_copy = params.copy()
            del params_copy["RUN_ID"]
        else:
            params_copy = params # 如果没有RUN_ID，直接使用原字典

        for key, value in params_copy.items():
            f.write(f"{key}: {value}\n")
    print(f"💾 超参数已保存至: {filepath}")


if __name__ == '__main__':
    # =========================================================
    # 定义本次训练的唯一标识符和根保存目录
    evaluate = True # **设置为 True 进行评估，False 进行训练**

    if evaluate:
        # !!! 在这里，你需要手动更改为你想要评估的训练运行的目录名 !!!
        # 例如，如果你之前训练时生成了一个名为 'runs/20231026-153045' 的目录，就填这个
        desired_timestamp = '20231027-143000' # <--- 修改为你要评估的模型的 timestamp
        run_id = desired_timestamp # 用于记录当前评估运行的ID
        base_save_dir = f'runs/{run_id}'
        print(f"---- 评估模式，加载模型来自: {base_save_dir} ----")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = timestamp # 用于记录当前训练运行的ID
        base_save_dir = f'runs/{run_id}'
        print(f"---- 训练模式开始，结果保存至: {base_save_dir} ----")

    # 基于唯一标识符构建检查点和图表的具体路径
    chkpt_base_dir = os.path.join(base_save_dir, 'checkpoints')
    plot_base_dir = os.path.join(base_save_dir, 'plots')
    image_base_dir = os.path.join(base_save_dir, 'images') # 评估模式下保存图片

    # =========================================================

    # 初始化环境
    env = UAVEnv()
    n_agents = env.num_agents

    # 获取每个智能体的观测维度
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])

    # Critic 网络的输入维度是所有智能体观测维度的总和
    critic_dims = sum(actor_dims)

    n_actions = 2 # 每个智能体的动作维度

    # 定义学习率常量
    ALPHA = 0.0001
    BETA = 0.001#0.003

    # 定义网络隐藏层维度
    FC1_DIMS = 128
    FC2_DIMS = 128

    # 经验回放缓冲区大小
    MEMORY_SIZE = 1000000
    BATCH_SIZE = 256

    # 训练/评估参数
    PRINT_INTERVAL = 100 # 每隔多少回合打印一次信息
    N_GAMES = 7500 # 总训练或评估回合数
    MAX_STEPS = 100 # 每个回合的最大步数

    # =========================================================
    # 收集超参数并保存
    hyperparameters = {
        "RUN_ID": run_id, # 将当前运行的ID添加到超参数字典的第一项
        "N_AGENTS": n_agents,
        "ACTOR_DIMS": actor_dims,
        "CRITIC_DIMS": critic_dims,
        "N_ACTIONS_PER_AGENT": n_actions,
        "ACTOR_LR": ALPHA,
        "CRITIC_LR": BETA,
        "FC1_DIMS": FC1_DIMS,
        "FC2_DIMS": FC2_DIMS,
        "MEMORY_SIZE": MEMORY_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "N_GAMES": N_GAMES,
        "MAX_STEPS_PER_EPISODE": MAX_STEPS,
        "PRINT_INTERVAL": PRINT_INTERVAL,
        "SCENARIO": "UAV_Round_up"
    }
    save_hyperparameters(plot_base_dir, hyperparameters) # 保存在 plots 目录下
    # =========================================================

    # 初始化 MADDPG 智能体
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=FC1_DIMS, fc2=FC2_DIMS,
                           alpha=ALPHA, beta=BETA, scenario='UAV_Round_up',
                           chkpt_dir=chkpt_base_dir)

    # 初始化经验回放缓冲区 (在评估模式下可以不需要，但保留无害)
    memory = MultiAgentReplayBuffer(MEMORY_SIZE, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=BATCH_SIZE)

    total_steps = 0 # 累计总训练步数

    # 历史记录列表
    score_history = []
    target_score_history = []
    actor_losses_history = []
    critic_losses_history = []

    successful_episodes = 0

    best_score = -30 # 用于保存最佳模型 (在评估模式下不使用此变量进行保存)

    program_start_time = time.time()

    # 模式设置和目录创建
    if evaluate:
        maddpg_agents.load_checkpoint() # 加载模型
        print('----  评估模式  ----')
        os.makedirs(image_base_dir, exist_ok=True)
        os.makedirs(plot_base_dir, exist_ok=True)
    else:
        print('----训练模式开始----')
        os.makedirs(plot_base_dir, exist_ok=True)

    # 使用 tqdm 创建进度条
    pbar = tqdm(range(N_GAMES), desc="进度", unit="回合")

    for i in pbar:
        obs = env.reset()
        obs = [np.array(o) if isinstance(o, list) else o for o in obs]

        score = 0
        score_target = 0
        dones = [False]*n_agents
        episode_step = 0

        current_episode_actor_losses = []
        current_episode_critic_losses = []

        episode_successful = False

        while not any(dones) and episode_step < MAX_STEPS:
            if evaluate:
                env_render = env.render()
                if episode_step % 10 == 0:
                    filename = f'episode_{i}_step_{episode_step}.png'
                    save_image(env_render, os.path.join(image_base_dir, filename))
                # time.sleep(0.01)

            actions = maddpg_agents.choose_action(obs, total_steps, evaluate)
            obs_, rewards, dones = env.step(actions)
            obs_ = [np.array(o) if isinstance(o, list) else o for o in obs_]

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS - 1:
                dones = [True]*n_agents

            if not evaluate:
                memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)
                if total_steps % 10 == 0:
                    actor_loss, critic_loss = maddpg_agents.learn(memory, total_steps)
                    if actor_loss is not None and critic_loss is not None:
                        current_episode_actor_losses.append(actor_loss)
                        current_episode_critic_losses.append(critic_loss)
                total_steps += 1

            score += sum(rewards[0:2])
            score_target += rewards[-1]

            obs = obs_
            episode_step += 1

        score_history.append(score)
        target_score_history.append(score_target)

        if not evaluate:
            if current_episode_actor_losses:
                actor_losses_history.append(np.mean(current_episode_actor_losses))
                critic_losses_history.append(np.mean(current_episode_critic_losses))
            else:
                actor_losses_history.append(None)
                critic_losses_history.append(None)

        if evaluate:
            if any(dones) and episode_step < MAX_STEPS:
                episode_successful = True
                successful_episodes += 1

        avg_score = np.mean(score_history[-100:]) if len(score_history) >= 100 else np.mean(score_history)
        avg_target_score = np.mean(target_score_history[-100:]) if len(target_score_history) >= 100 else np.mean(target_score_history)

        if not evaluate:
            if i % PRINT_INTERVAL == 0 and i > 0:
                if avg_score > best_score:
                    pbar.write(f'🔥 回合 {i}: 新的最佳追捕方平均得分 {avg_score:.1f} > 历史最佳 ({best_score:.1f}), 正在保存模型...')
                    maddpg_agents.save_checkpoint()
                    best_score = avg_score

            if i % PRINT_INTERVAL == 0 and i > 0:
                try:
                    current_alpha = maddpg_agents.agents[0].actor_optimizer.param_groups[0]['lr']
                    current_beta = maddpg_agents.agents[0].critic_optimizer.param_groups[0]['lr']
                except AttributeError:
                    current_alpha = ALPHA
                    current_beta = BETA

                display_actor_loss = f"{np.nanmean([l for l in actor_losses_history[-PRINT_INTERVAL:] if l is not None]):.4f}" if any(l is not None for l in actor_losses_history[-PRINT_INTERVAL:]) else "N/A"
                display_critic_loss = f"{np.nanmean([l for l in critic_losses_history[-PRINT_INTERVAL:] if l is not None]):.4f}" if any(l is not None for l in critic_losses_history[-PRINT_INTERVAL:]) else "N/A"

                pbar.set_postfix({
                    '追捕方平均分': f'{avg_score:.1f}',
                    '目标方平均分': f'{avg_target_score:.1f}',
                    '最佳追捕方平均': f'{best_score:.1f}',
                    '总步数': total_steps,
                    'Actor Loss': display_actor_loss,
                    'Critic Loss': display_critic_loss
                })
                pbar.write(f'\n回合 {i}, '
                           f'当前追捕方总奖励: {score:.1f}, 当前目标方总奖励: {score_target:.1f} | '
                           f'追捕方平均得分: {avg_score:.1f}, 目标方平均得分: {avg_target_score:.1f} | '
                           f'Alpha: {current_alpha:.6f}, Beta: {current_beta:.6f} | '
                           f'平均 Actor Loss: {display_actor_loss}, 平均 Critic Loss: {display_critic_loss}')
        else: # 评估模式下的日志
            pbar.set_postfix({
                '当前追捕方得分': f'{score:.1f}',
                '当前目标方得分': f'{score_target:.1f}',
                '追捕方平均分': f'{avg_score:.1f}',
                '目标方平均分': f'{avg_target_score:.1f}',
                '成功': '是' if episode_successful else '否'
            })
            if i % PRINT_INTERVAL == 0 and i > 0:
                pbar.write(f'评估回合 {i}, '
                           f'追捕方总得分: {score:.1f}, 目标方总得分: {score_target:.1f} | '
                           f'回合成功: {"是" if episode_successful else "否"}')

    pbar.close()

    # --- 运行后操作：保存数据和绘制图表 ---
    print('\n---- 运行结束 ----')

    score_df = pd.DataFrame({
        '回合': range(len(score_history)),
        '追捕方得分': score_history,
        '目标方得分': target_score_history
    })

    if evaluate:
        score_csv_path = os.path.join(plot_base_dir, 'evaluation_scores.csv')
        print('💾 正在保存评估得分数据...')
    else:
        score_csv_path = os.path.join(plot_base_dir, 'training_scores.csv')
        print('💾 正在保存训练得分数据...')

        loss_df = pd.DataFrame({
            '回合': range(len(actor_losses_history)),
            'Actor_Loss': actor_losses_history,
            'Critic_Loss': critic_losses_history
        })
        loss_csv_path = os.path.join(plot_base_dir, 'training_losses.csv')
        print('💾 正在保存训练损失数据...')
        if not os.path.exists(loss_csv_path):
            loss_df.to_csv(loss_csv_path, header=True, index=False)
        else:
            loss_df.to_csv(loss_csv_path, mode='a', header=False, index=False)

    if not os.path.exists(score_csv_path):
        score_df.to_csv(score_csv_path, header=True, index=False)
    else:
        score_df.to_csv(score_csv_path, mode='a', header=False, index=False)

    if evaluate:
        print('📈 正在绘制最终评估曲线...')
        plot_curves(score_history, target_score_history, plot_base_dir, filename='evaluation_curves.png', title_prefix="评估")
    else:
        print('📈 正在绘制最终训练曲线...')
        plot_curves(score_history, target_score_history, plot_base_dir, filename='training_curves.png', title_prefix="训练")

        if any(l is not None for l in actor_losses_history):
            plot_losses(actor_losses_history, critic_losses_history, plot_base_dir, filename='training_losses.png')
            print('📈 正在绘制最终训练损失曲线...')

    total_elapsed_time = time.time() - program_start_time
    print(f'✅ 总运行时间: {format_time(total_elapsed_time)}')
    if not evaluate:
        print(f'训练期间达到的最佳追捕方平均得分: {best_score:.1f}')
    else:
        success_rate = (successful_episodes / N_GAMES) * 100 if N_GAMES > 0 else 0
        print(f'在 {N_GAMES} 个评估回合中，追捕方平均得分: {np.mean(score_history):.1f}')
        print(f'在 {N_GAMES} 个评估回合中，目标方平均得分: {np.mean(target_score_history):.1f}')
        print(f'评估成功率: {success_rate:.2f}% ({successful_episodes}/{N_GAMES} 回合成功)')