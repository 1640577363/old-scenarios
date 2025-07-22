import numpy as np


class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size):
        '''

        :param max_size:缓冲区最大容量
        :param critic_dims:Critic网络输入维度（全局状态）
        :param actor_dims:列表，包含每个智能体的观测维度
        :param n_actions:每个智能体的动作数量
        :param n_agents:智能体总数
        :param batch_size:采样批次大小
        '''
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        # 当前全局状态
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        # 下一时刻全局状态
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))

        # 每个智能体的奖励
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        # 终止标志（每个智能体独立）
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        '''
        actor_state_memory：当前局部观测
        actor_new_state_memory：下一时刻局部观测
        actor_action_memory：执行的动作
        '''
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            #为每个智能体初始化numpy数组
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_, done):
        # this introduces a bug: if we fill up the memory capacity and then
        # zero out our actor memory, the critic will still have memories to access
        # while the actor will have nothing but zeros to sample. Obviously
        # not what we intend.
        # In reality, there's no problem with just using the same index
        # for both the actor and critic states. I'm not sure why I thought
        # this was necessary in the first place. Sorry for the confusion!

        # if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()
# 注释中提到的bug已修复：原代码在缓冲区满时会错误地重新初始化Actor记忆，
# 现已移除该问题代码。当前实现使用统一的索引管理所有存储空间。

        '''
        raw_obs：智能体的当前局部观测（列表）
        state：当前全局状态
        action：智能体执行的动作（列表）
        reward：奖励值（列表）
        raw_obs_：下一时刻局部观测
        state_：下一时刻全局状态
        done：终止标志（列表）
        '''

#       计算存储位置

        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            '''当前局部观测
                下一时刻局部观测
                执行的动作'''
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]


        '''存储Critic网络所需的全局信息：
        当前全局状态
        下一时刻全局状态
        奖励值
        终止标志'''
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        #更新存储计数器
        self.mem_cntr += 1

    def sample_buffer(self):
        #确定当前有效记忆量
        max_mem = min(self.mem_cntr, self.mem_size)
        #随机选择批次的索引（不重复采样）
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        #随机采样全局经验
        '''
        随机采样Critic网络所需数据：
        全局状态
        奖励
        下一时刻全局状态
        终止标志'''
        # 批量采样
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        # 随机采样局部经验
#为每个智能体采样Actor网络所需数据
        actor_states = []
        actor_new_states = []
        actions = []

        #当前局部观测
        # 下一时刻局部观测
        # 执行的动作
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
            actor_new_states, states_, terminal

    def ready(self):
#         检查缓冲区是否有足够数据采样
#          当存储量 ≥ 批次大小时返回True
        if self.mem_cntr >= self.batch_size:
            return True
