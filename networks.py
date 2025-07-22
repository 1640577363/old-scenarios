import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        '''
        :param beta:学习率
        :param input_dims:状态输入维度（多个智能体观测拼接而成的状态）
        :param fc1_dims:
        :param fc2_dims:隐藏层维度
        :param n_agents:
        :param n_actions:动作维度（多个智能体动作拼接）
        :param name:
        :param chkpt_dir:用于保存模型的路径名
        '''
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        #input_dims:所有智能体观测拼接在一起的维度，也就是 全局状态维度
        # n_agents * n_actions：所有智能体动作拼接在一起的维度，代表整个系统联合动作；
        #当前所有智能体的状态 + 所有智能体的动作”，
        # 输出：
        # “这些行为组合在当前环境下的好坏评估值（Q 值）”。
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        #创建一个 Adam 优化器，来更新该网络中的所有参数,
        # lr=beta 指的是学习率（learning rate），这是模型在每一步更新时“走多远”的尺度。
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        #这是一个学习率调度器，用于在训练过程中逐步减小学习率，以便更稳定地收敛
        #step_size:每过 5000 次调用 scheduler.step() 后，调整一次学习率,每次乘以gamma（0.33）
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)    
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        '''

        :param alpha:学习率
        :param input_dims: 本智能体的观测维度
        :param fc1_dims:
        :param fc2_dims:
        :param n_actions:本智能体输出的动作维度
        :param name:
        :param chkpt_dir:
        '''
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

#这是标准的三层 MLP，最后输出动作
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        #F.leaky_relu 是 PyTorch 中的激活函数之一， 全称 Leaky Rectified Linear Unit，中文可称为“带泄露的线性整流函数”
        #正数保留、负数缩小1000倍
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))

        #进行 Softsign 激活处理，将数值控制在-1——1
        pi = nn.Softsign()(self.pi(x)) # [-1,1](self.pi(x))是未归一化的线性输出，值域是 (-∞, +∞)

        return pi

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

