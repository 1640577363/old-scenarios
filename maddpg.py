import os
import torch as T
import torch.nn.functional as F
from agent import Agent
from torch.utils.tensorboard import SummaryWriter


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.02, fc1=128,
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        '''
        :param actor_dims: 每个智能体的 Actor 网络输入维度列表
        :param critic_dims: Critic 网络的输入维度（全局状态）
        :param n_agents:
        :param n_actions:每个智能体的动作维度
        :param scenario:场景名称（默认 'simple'）
        :param alpha:Actor 学习率（默认 0.01）
        :param beta:Critic 学习率（默认 0.02）
        :param fc1:神经网络第一隐藏层大小（默认 128）
        :param fc2:神经网络第二隐藏层大小（默认 128）
        :param gamma:折扣因子（默认 0.99）
        :param tau:目标网络软更新系数（默认 0.01）
        :param chkpt_dir:模型保存目录（默认 'tmp/maddpg/'）
        '''

        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario  #路径tmp/maddpg/UAV_Round_up

        # 可用于训练可视化
        self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir))

    def save_checkpoint(self,suffix=""):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        #遍历所有智能体，调用加载模型方法
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()  #如何加载？

    def choose_action(self, raw_obs, time_step, evaluate):  # timestep for exploration
        '''

        :param raw_obs: 原始观测列表（每个元素对应一个智能体）
        :param time_step:当前时间步（用于探索率计算
        :param evaluate:是否评估模式（True 时不探索）
        :return:返回所有智能体的动作列表
        '''
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], time_step, evaluate)
            actions.append(action)
        return actions

    def learn(self, memory, total_steps):
        #方法判断缓冲区是否达到批处理大小（batch_size），避免在样本不足时更新网络
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones = memory.sample_buffer()
                                            #dones的内容是
        '''
        actor_states：各智能体的当前状态（列表）
        states：全局状态
        actions：各智能体执行的动作
        rewards：奖励
        actor_new_states：各智能体的下一状态
        states_：全局下一状态
        dones：终止标志
        '''

        #获取第一个智能体的设备（CPU/GPU），
        # 确保所有张量都在相同设备上计算
        device = self.agents[0].actor.device

        #将采样数据转换为 PyTorch 张量并移到相应设备
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        #存储各智能体在下一状态的目标动作
        all_agents_new_actions = []
        # 存储当前状态各智能体的动作
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):  #遍历每个智能体
            # 当前智能体在下一时刻的状态
            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)
            #           使用目标Actor网络预测的下一状态动作
            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)  #将预测动作添加到目标动作列表
            # 保存当前状态的实际动作
            old_agents_actions.append(actions[agent_idx])

        #下一状态的联合动作
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        #T.cat：拼接各智能体动作
        #为什么要拼接？见思维导图

        #当前状态的联合动作
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():  #禁用梯度计算
                #计算目标Critic值
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                                        #states_是状态 ，new_actions是动作预测
                # 目标价值 = 即时奖励 + γ * 下一状态价值（如果未终止
                target = rewards[:, agent_idx] + (1 - dones[:, 0].int()) * agent.gamma * critic_value_
                #代码解释?见思维导图

            # target_critic和critic的区别

            #计算当前 Critic 的价值估计
            critic_value = agent.critic.forward(states, old_actions).flatten()

            #计算 MSE 损失（目标价值 vs 当前估计）
            critic_loss = F.mse_loss(target, critic_value)

            #梯度清零 → 反向传播 → 优化器更新 → 学习率调度器更新
            agent.critic.optimizer.zero_grad()  #清空梯度

            #什么是正向传播和反向传播
            critic_loss.backward(retain_graph=True)  # 反向传播，保留计算图
            agent.critic.optimizer.step()  #更新网络参数
            agent.critic.scheduler.step()  #更新学习率

            #?
            # 当前智能体的状态
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)

            oa = old_actions.clone()
            #修改联合动作中当前智能体的部分
            oa[:, agent_idx * self.n_actions:agent_idx * self.n_actions + self.n_actions] = agent.actor.forward(
                mu_states)#替换所有行的（n_actions=2时为2-3列）2-3列


            #？
            #最大化Critic评价（负号使梯度上升）
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            #-T.mean()计算批量样本的平均价值,因为 PyTorch 默认最小化损失，而我们需要最大化价值


            #更新Actor网络（类似Critic更新流程）
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            '''
            Actor 更新：

            获取当前智能体的状态
            
            克隆联合动作，仅替换当前智能体的动作为 Actor 网络输出
            
            计算 Actor 损失：最大化 Critic 对修改后动作的评价
            
            梯度清零 → 反向传播 → 优化器更新 → 学习率调度器更新
            '''

            # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
            # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)

            # for name, param in agent.actor.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
            # for name, param in agent.critic.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)

        for agent in self.agents:
            agent.update_network_parameters()
            # 所有智能体更新目标网络参数（软更新）
            # 使用软更新公式：θ_target = τ*θ + (1-τ)*θ_target
