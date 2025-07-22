import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.0001, beta=0.0002, fc1=128,
                    fc2=128, gamma=0.99, tau=0.01):
        '''
            actor_dims: 当前智能体观测维度
            critic_dims: Critic 网络输入维度（全局状态维度）
            n_actions: 单智能体动作维度
            n_agents: 智能体总数，用于 Critic 网络拼接动作
            agent_idx: 智能体索引，用于命名和动作切片
            chkpt_dir: 模型保存目录
            alpha, beta: Actor、Critic 的学习率
            fc1, fc2: 两层隐藏层神经元数
            gamma: 折扣因子
            tau: 软更新系数        '''
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx#?

        #该网络负责在执行阶段（或训练阶段的策略梯度更新时）根据当前观测输出动作。
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        #Critic 在训练时接收全局状态与所有智能体动作，输出 Q 值估计。
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')

        #参数同在线网络，仅名称和保存文件区分。
        # 目标网络用于生成稳定的目标值，防止训练不稳定。
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, time_step, evaluate=False):
        '''

        :param observation:
        :param time_step:
        :param evaluate:
        :return:
        '''
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        #通过 Actor 网络前向传播，得到无噪声的动作张量 actions，形状 [1, n_actions]。
        actions = self.actor.forward(state)

        # exploration
        '''
        max_noise：初始噪声幅度

        min_noise：噪声下限
        
        decay_rate：每个时间步噪声衰减系数
        '''
        max_noise = 0.75
        min_noise = 0.01
        decay_rate = 0.999995

        # 计算当前步的噪声幅度
        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        #生成一个形状 [n_actions]、在 [-1,1) 区间均匀分布的随机噪声。
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1 # [-1,1)

        #在训练模式（evaluate=False）下，将噪声按 noise_scale 缩放后加入动作；
        if not evaluate:
            noise = noise_scale * noise
        else:
            # 评估模式下置零，不加噪声。
            noise = 0 * noise
        
        action = actions + noise
        action_np = action.detach().cpu().numpy()[0]
        magnitude = np.linalg.norm(action_np)

        if magnitude > 0.04:#防止动作值过大导致训练不稳定
            action_np = action_np / magnitude * 0.04
        return action_np

    # 实现了MADDPG算法中目标网络的软更新(Polyak Averaging)机制
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau#更新系数

        #目标Actor网络更新
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        # 逐参数融合更新
        #
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        # # 内存高效方案  DeepSeek提供
        # with T.no_grad():
        #     for online_param, target_param in zip(self.actor.parameters(),
        #                                           self.target_actor.parameters()):
        #         target_param.data.copy_(tau*online_param.data + (1-tau)*target_param.data)

        self.target_actor.load_state_dict(actor_state_dict)

        #目标Critic网络更新
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()
            #

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
