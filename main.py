import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from tqdm import tqdm

# 导入你的自定义模块
from maddpg import MADDPG
from sim_env import UAVEnv
from buffer import MultiAgentReplayBuffer

warnings.filterwarnings('ignore')

def obs_list_to_state_vector(obs):
    """将观测列表展平为单一状态向量。"""
    # 确保每个观测都是可展平的（例如，NumPy 数组）
    state = np.hstack([np.ravel(o) for o in obs])
    return state

def save_image(env_render, filename):
    """保存环境渲染图像。"""
    # env.render() 应该返回一个 RGBA 格式的 numpy 数组
    image = Image.fromarray(env_render, 'RGBA')
    image = image.convert('RGB') # 转换为 RGB 以便更广泛兼容
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

    # 过滤掉 None 值以避免绘图错误
    plot_actor_losses = [l for l in actor_losses if l is not None]
    plot_critic_losses = [l for l in critic_losses if l is not None]
    # plot_episodes 应该是与损失数据点对应的回合索引
    # 由于损失是按回合记录的，且每个回合可能包含多次学习，这里记录的是回合的平均损失
    # 所以直接使用过滤后的列表长度作为索引
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

if __name__ == '__main__':
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
    BETA = 0.003

    # 初始化 MADDPG 智能体
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128,
                           alpha=ALPHA, beta=BETA, scenario='UAV_Round_up', # 确保 scenario 参数匹配你的 chkpt_dir
                           chkpt_dir='tmp/maddpg/')

    # 初始化经验回放缓冲区
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=256)

    # 训练/评估参数
    PRINT_INTERVAL = 100 # 每隔多少回合打印一次信息
    N_GAMES = 15000 # 总训练或评估回合数
    MAX_STEPS = 100 # 每个回合的最大步数
    total_steps = 0 # 累计总训练步数

    # 历史记录列表
    score_history = [] # 追捕方得分历史
    target_score_history = [] # 目标方得分历史
    actor_losses_history = [] # Actor 损失历史 (每回合平均)
    critic_losses_history = [] # Critic 损失历史 (每回合平均)

    successful_episodes = 0 # 用于统计评估模式下的成功回合数

    evaluate = False # **设置为 True 进行评估，False 进行训练**
    best_score = -30 # 用于保存最佳模型

    program_start_time = time.time()

    # 模式设置和目录创建
    if evaluate:
        maddpg_agents.load_checkpoint() # 评估模式下加载已训练模型
        print('----  评估模式  ----')
        os.makedirs('evaluation_images', exist_ok=True)
        os.makedirs('evaluation_plots', exist_ok=True)
    else:
        print('----训练模式开始----')
        os.makedirs('training_plots', exist_ok=True) # 确保训练图表目录存在

    # 使用 tqdm 创建进度条
    pbar = tqdm(range(N_GAMES), desc="进度", unit="回合")

    for i in pbar: # 遍历每个回合
        obs = env.reset() # 重置环境，获取初始观测
        # 确保观测是 NumPy 数组
        obs = [np.array(o) if isinstance(o, list) else o for o in obs]

        score = 0 # 当前回合追捕方总奖励
        score_target = 0 # 当前回合目标方总奖励
        dones = [False]*n_agents # 智能体终止标志
        episode_step = 0 # 当前回合的步数

        current_episode_actor_losses = [] # 存储当前回合中每个学习步的 Actor 损失
        current_episode_critic_losses = [] # 存储当前回合中每个学习步的 Critic 损失

        episode_successful = False # 标记当前回合是否成功（仅用于评估）

        # 回合循环
        while not any(dones) and episode_step < MAX_STEPS:
            if evaluate:
                env_render = env.render() # 渲染环境
                if episode_step % 10 == 0: # 每 10 步保存一次评估图像
                    filename = f'evaluation_images/episode_{i}_step_{episode_step}.png'
                    save_image(env_render, filename)
                # time.sleep(0.01) # 可选：减慢可视化速度

            # 智能体选择动作
            actions = maddpg_agents.choose_action(obs, total_steps, evaluate)
            # 环境执行动作，返回新的观测、奖励和终止标志
            obs_, rewards, dones = env.step(actions)

            # 确保新的观测也是 NumPy 数组
            obs_ = [np.array(o) if isinstance(o, list) else o for o in obs_]

            # 将智能体观测组合成全局状态
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            # 如果达到最大步数，强制结束回合
            if episode_step >= MAX_STEPS - 1: # -1 是因为 episode_step 在循环末尾还会 +1
                dones = [True]*n_agents

            if not evaluate: # **仅在训练模式下进行经验存储和学习**
                memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)
                if total_steps % 10 == 0: # 每 10 个总步数进行一次学习
                    # 从 MADDPG 模型的 learn 方法获取损失
                    actor_loss, critic_loss = maddpg_agents.learn(memory, total_steps)
                    if actor_loss is not None and critic_loss is not None:
                        current_episode_actor_losses.append(actor_loss)
                        current_episode_critic_losses.append(critic_loss)
                total_steps += 1 # 总步数只在训练模式下递增

            # 累加当前回合的奖励
            score += sum(rewards[0:2]) # 追捕方总奖励（假设前两个是追捕方）
            score_target += rewards[-1] # 目标方总奖励（假设最后一个是目标方）

            obs = obs_ # 更新观测
            episode_step += 1 # 回合步数递增

        # 回合结束后的处理
        score_history.append(score)
        target_score_history.append(score_target)

        # 记录每回合的平均损失 (仅在训练模式下)
        if not evaluate:
            if current_episode_actor_losses: # 如果有学习步骤产生损失
                actor_losses_history.append(np.mean(current_episode_actor_losses))
                critic_losses_history.append(np.mean(current_episode_critic_losses))
            else: # 如果没有学习（例如缓冲区未满），则记录 None
                actor_losses_history.append(None)
                critic_losses_history.append(None)

        # 判断当前回合是否成功（仅在评估模式下统计）
        if evaluate:
            # 你需要根据你的环境定义更精确的成功标准。
            # 这里简化为：如果回合在达到 MAX_STEPS 之前因某个智能体 'done' 而结束，则认为是成功。
            # 更精确的判断可能需要检查 `dones` 列表中具体哪个智能体 `done` 了，例如：
            # if dones[索引_目标智能体] and episode_step < MAX_STEPS: # 如果目标智能体被捕获
            if any(dones) and episode_step < MAX_STEPS:
                episode_successful = True
                successful_episodes += 1

        # 计算过去 100 回合的平均得分
        avg_score = np.mean(score_history[-100:]) if len(score_history) >= 100 else np.mean(score_history)
        avg_target_score = np.mean(target_score_history[-100:]) if len(target_score_history) >= 100 else np.mean(target_score_history)

        # 进度条和日志更新
        if not evaluate: # 训练模式下的日志
            if i % PRINT_INTERVAL == 0 and i > 0:
                # 检查并保存最佳模型
                if avg_score > best_score:
                    pbar.write(f'🔥 回合 {i}: 新的最佳追捕方平均得分 {avg_score:.1f} > 历史最佳 ({best_score:.1f}), 正在保存模型...')
                    maddpg_agents.save_checkpoint()
                    best_score = avg_score

            if i % PRINT_INTERVAL == 0 and i > 0:
                # 获取当前学习率
                try:
                    current_alpha = maddpg_agents.agents[0].actor_optimizer.param_groups[0]['lr']
                    current_beta = maddpg_agents.agents[0].critic_optimizer.param_groups[0]['lr']
                except AttributeError: # 如果无法通过这种方式获取，则使用初始定义的常量
                    current_alpha = ALPHA
                    current_beta = BETA

                # 计算并显示过去 PRINT_INTERVAL 回合的平均损失
                # np.nanmean 会忽略 None 值，这在缓冲区未满时很有用
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

    pbar.close() # 关闭进度条

    # --- 运行后操作：保存数据和绘制图表 ---
    print('\n---- 运行结束 ----')

    # 保存得分历史到 CSV
    score_df = pd.DataFrame({
        '回合': range(len(score_history)),
        '追捕方得分': score_history,
        '目标方得分': target_score_history
    })

    if evaluate:
        score_csv_path = 'evaluation_scores.csv'
        print('💾 正在保存评估得分数据...')
    else:
        score_csv_path = 'training_scores.csv'
        print('💾 正在保存训练得分数据...')

        # 保存损失历史到 CSV (仅在训练模式下)
        loss_df = pd.DataFrame({
            '回合': range(len(actor_losses_history)),
            'Actor_Loss': actor_losses_history,
            'Critic_Loss': critic_losses_history
        })
        loss_csv_path = 'training_losses.csv'
        print('💾 正在保存训练损失数据...')
        # 检查文件是否存在，决定是创建新文件还是追加
        if not os.path.exists(loss_csv_path):
            loss_df.to_csv(loss_csv_path, header=True, index=False)
        else:
            # 如果文件已存在，则以追加模式写入，并跳过头部
            loss_df.to_csv(loss_csv_path, mode='a', header=False, index=False)

    # 检查文件是否存在，决定是创建新文件还是追加
    if not os.path.exists(score_csv_path):
        score_df.to_csv(score_csv_path, header=True, index=False)
    else:
        score_df.to_csv(score_csv_path, mode='a', header=False, index=False)

        # 绘制曲线图
    if evaluate:
        print('📈 正在绘制最终评估曲线...')
        plot_curves(score_history, target_score_history, 'evaluation_plots', filename='evaluation_curves.png', title_prefix="评估")
    else:
        print('📈 正在绘制最终训练曲线...')
        plot_curves(score_history, target_score_history, 'training_plots', filename='training_curves.png', title_prefix="训练")

        # 额外绘制损失曲线 (只在训练模式下且有损失数据时)
        if any(l is not None for l in actor_losses_history):
            plot_losses(actor_losses_history, critic_losses_history, 'training_plots', filename='training_losses.png')
            print('📈 正在绘制最终训练损失曲线...')

    # 最终运行时间与性能总结
    total_elapsed_time = time.time() - program_start_time
    print(f'✅ 总运行时间: {format_time(total_elapsed_time)}')
    if not evaluate:
        print(f'训练期间达到的最佳追捕方平均得分: {best_score:.1f}')
    else:
        success_rate = (successful_episodes / N_GAMES) * 100 if N_GAMES > 0 else 0
        print(f'在 {N_GAMES} 个评估回合中，追捕方平均得分: {np.mean(score_history):.1f}')
        print(f'在 {N_GAMES} 个评估回合中，目标方平均得分: {np.mean(target_score_history):.1f}')
        print(f'评估成功率: {success_rate:.2f}% ({successful_episodes}/{N_GAMES} 回合成功)')