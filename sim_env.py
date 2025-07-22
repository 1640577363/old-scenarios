import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy

class UAVEnv:
    def __init__(self,length=2,num_obstacle=3,num_agents=4):
        self.length = length # length of boundary
        self.num_obstacle = num_obstacle # number of obstacles
        self.num_agents = num_agents
        self.time_step = 0.5 # update time step
        #速度限制
        self.v_max = 0.1
        self.v_max_e = 0.12
        #加速度限制
        self.a_max = 0.04
        self.a_max_e = 0.05
        # 传感器参数
        self.L_sensor = 0.2
        self.num_lasers = 16 # num of laserbeams
        # 初始化激光雷达读数（默认最大距离）
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]

        self.agents = ['agent_0','agent_1','agent_2','target']
        self.info = np.random.get_state() # get seed
        # 创建障碍物
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]
        # 历史位置记录
        self.history_positions = [[] for _ in range(num_agents)]
        # 定义动作空间（每个智能体的加速度向量）
        self.action_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
            } # action represents [a_x,a_y]
        # 定义观测空间
        self.observation_space = {
            'agent_0': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_1': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'agent_2': spaces.Box(low=-np.inf, high=np.inf, shape=(26,)),
            'target': spaces.Box(low=-np.inf, high=np.inf, shape=(23,))
        }
        

    def reset(self):
        SEED = random.randint(1,1000)
        random.seed(SEED)
        # 重置位置和速度
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            if i != self.num_agents - 1: # if not target # 围捕无人机
                # 随机初始位置（左下角区域）
                self.multi_current_pos.append(np.random.uniform(low=0.1,high=0.4,size=(2,)))
            else: # for target 目标（固定初始位置）
                # self.multi_current_pos.append(np.array([1.0,0.25]))
                self.multi_current_pos.append(np.array([0.5,1.75]))
                # 初始速度为零
            self.multi_current_vel.append(np.zeros(2)) # initial velocity = [0,0]

        # update lasers更新激光雷达读数
        self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        # 获取多智能体观测
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self,actions):
        last_d2target = [] # 记录上一步到目标的距离
        # print(actions)
        # time.sleep(0.1)

        # 更新每个智能体的状态
        for i in range(self.num_agents):

            pos = self.multi_current_pos[i]
            if i != self.num_agents - 1:
                pos_taget = self.multi_current_pos[-1]
                # 记录围捕无人机到目标的距离
                last_d2target.append(np.linalg.norm(pos-pos_taget))

            # 更新速度（加速度×时间步长）
            self.multi_current_vel[i][0] += actions[i][0] * self.time_step
            self.multi_current_vel[i][1] += actions[i][1] * self.time_step
            # 速度限幅
            vel_magnitude = np.linalg.norm(self.multi_current_vel)
            if i != self.num_agents - 1:# 围捕无人机
                if vel_magnitude >= self.v_max:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            else: # 目标
                if vel_magnitude >= self.v_max_e:
                    self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max_e

            # 更新位置（速度×时间步长）
            self.multi_current_pos[i][0] += self.multi_current_vel[i][0] * self.time_step
            self.multi_current_pos[i][1] += self.multi_current_vel[i][1] * self.time_step

        # Update obstacle positions 更新障碍物位置
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            # Check for boundary collisions and adjust velocities
            # 边界碰撞检测（反弹）
            for dim in [0, 1]:# x和y方向
                if obs.position[dim] - obs.radius < 0:# 左/下边界
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1   # 速度反向
                elif obs.position[dim] + obs.radius > self.length:# 右/上边界
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1# 速度反向
        # 更新激光雷达并检测碰撞
        Collided = self.update_lasers_isCollied_wrapper()
        # 计算奖励和终止条件
        rewards, dones= self.cal_rewards_dones(Collided,last_d2target)
        # 获取新的观测
        multi_next_obs = self.get_multi_obs()
        # sequence above can't be disrupted

        return multi_next_obs, rewards, dones

    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs

    # 获取多智能体观测
    def get_multi_obs(self):
        total_obs = [] # 所有智能体的观测
        single_obs = []# 单个智能体的观测
        S_evade_d = [] # dim 3 only for target # 目标专用的逃避距离
        for i in range(self.num_agents):
            # 自身状态（位置和速度，归一化）
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]

            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ] # dim 4

            S_team = [] # dim 4 for 3 agents 1 target   # 队友位置
            S_target = [] # dim 2 # 目标信息
            for j in range(self.num_agents):
                if j != i and j != self.num_agents - 1: # 其他围捕无人机
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0]/self.length,pos_other[1]/self.length])
                elif j == self.num_agents - 1:# 目标
                    # 计算到目标的距离和角度
                    pos_target = self.multi_current_pos[j]

                    d = np.linalg.norm(pos - pos_target)
                    theta = np.arctan2(pos_target[1]-pos[1], pos_target[0]-pos[0])
                    S_target.extend([d/np.linalg.norm(2*self.length), theta])# 归一化距离
                    if i != self.num_agents - 1:
                        S_evade_d.append(d/np.linalg.norm(2*self.length))# 用于目标的观测
            # 激光雷达读数
            S_obser = self.multi_current_lasers[i] # dim 16

            # 组装观测
            if i != self.num_agents - 1:
                # [自身状态(4) + 队友位置(4) + 激光雷达(16) + 目标信息(2)] = 26维
                single_obs = [S_uavi,S_team,S_obser,S_target]
            else:
                # [自身状态(4) + 激光雷达(16) + 围捕距离(3)] = 23维
                single_obs = [S_uavi,S_obser,S_evade_d]

            # 展平嵌套列表
            _single_obs = list(itertools.chain(*single_obs))
            total_obs.append(_single_obs)
            
        return total_obs

    def cal_rewards_dones(self,IsCollied,last_d):
        dones = [False] * self.num_agents# 终止标志
        rewards = np.zeros(self.num_agents)# 终止标志
        mu1 = 0.7 # r_near 接近奖励系数
        mu2 = 0.4 # r_safe安全奖励系数
        mu3 = 0.01 # r_multi_stage 多阶段任务奖励系数
        mu4 = 5 # r_finish 完成任务奖励系数
        d_capture = 0.3 #捕获距离
        d_limit = 0.75 #距离阈值

        ## 1 reward for single rounding-up-UAVs:
        #接近目标奖励（围捕无人机）
        for i in range(3):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            pos_target = self.multi_current_pos[-1]
            v_i = np.linalg.norm(vel)
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec) # distance to target

            # 计算速度方向与目标方向的夹角余弦
            cos_v_d = np.dot(vel,dire_vec)/(v_i*d + 1e-3)
            # 接近奖励：速度越大且方向越对准目标，奖励越高
            r_near = abs(2*v_i/self.v_max)*cos_v_d
            # r_near = min(abs(v_i/self.v_max)*1.0/(d + 1e-5),10)/5
            rewards[i] += mu1 * r_near # TODO: if not get nearer then receive negative reward
        
        ## 2 collision reward for all UAVs:安全奖励（所有智能体）
        for i in range(self.num_agents):# 发生碰撞
            if IsCollied[i]:
                r_safe = -10
            else:
                # 安全奖励：最近的激光距离越近，奖励越低
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor
            rewards[i] += mu2 * r_safe

        ## 3 multi-stage's reward for rounding-up-UAVs
        # 多阶段任务奖励
        p0 = self.multi_current_pos[0]
        p1 = self.multi_current_pos[1]
        p2 = self.multi_current_pos[2]
        pe = self.multi_current_pos[-1]
        S1 = cal_triangle_S(p0,p1,pe)
        S2 = cal_triangle_S(p1,p2,pe)
        S3 = cal_triangle_S(p2,p0,pe)
        S4 = cal_triangle_S(p0,p1,p2)
        d1 = np.linalg.norm(p0-pe)
        d2 = np.linalg.norm(p1-pe)
        d3 = np.linalg.norm(p2-pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)
        # 3.1 reward for target UAV:
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d),-2,2)
        # print(rewards[-1])
        # 3.2 stage-1 track
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= d_capture for d in [d1, d2, d3]):
            r_track = - Sum_d/max([d1,d2,d3])
            rewards[0:2] += mu3*r_track
        # 3.3 stage-2 encircle
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= d_capture for d in [d1, d2, d3])):
            r_encircle = -1/3*np.log(Sum_S - S4 + 1)
            rewards[0:2] += mu3*r_encircle
        # 3.4 stage-3 capture
        elif Sum_S == S4 and any(d > d_capture for d in [d1,d2,d3]):
            r_capture = np.exp((Sum_last_d - Sum_d)/(3*self.v_max))
            rewards[0:2] += mu3*r_capture
        
        ## 4 finish rewards
        if Sum_S == S4 and all(d <= d_capture for d in [d1,d2,d3]):
            rewards[0:2] += mu4*10
            dones = [True] * self.num_agents

        return rewards,dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos,obs_pos,r,self.L_sensor,self.num_lasers,self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    def render(self):

        plt.clf()
        
        # load UAV icon
        uav_icon = mpimg.imread('UAV.png')
        # icon_height, icon_width, _ = uav_icon.shape

        # plot round-up-UAVs
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            # Calculate the angle of the velocity vector
            angle = np.arctan2(vel[1], vel[0])

            # plt.scatter(pos[0], pos[1], c='b', label='hunter')
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            # plt.imshow(uav_icon, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            # plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            icon_size = 0.1  # Adjust this size to your icon's aspect ratio
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

            # # Visualize laser rays for each UAV(can be closed when unneeded)
            # lasers = self.multi_current_lasers[i]
            # angles = np.linspace(0, 2 * np.pi, len(lasers), endpoint=False)
            
            # for angle, laser_length in zip(angles, lasers):
            #     laser_end = np.array(pos) + np.array([laser_length * np.cos(angle), laser_length * np.sin(angle)])
            #     plt.plot([pos[0], laser_end[0]], [pos[1], laser_end[1]], 'b-', alpha=0.2)

        # plot target
        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length+0.1)
        plt.ylim(-0.1, self.length+0.1)
        plt.draw()
        plt.legend()
        # plt.pause(0.01)
        # Save the current figure to a buffer
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        
        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        plt.clf()
        
        uav_icon = mpimg.imread('UAV.png')

        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)
            
            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], color=color, alpha=0.7)
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=1)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

        plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        pos_e = copy.deepcopy(self.multi_current_pos[-1])
        self.history_positions[-1].append(pos_e)
        trajectory = np.array(self.history_positions[-1])
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

    def close(self):
        plt.close()

class obstacle():
    def __init__(self, length=2):
        self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        # speed = 0.03 
        speed = 0.02 # to make obstacle fixed
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        self.radius = np.random.uniform(0.1, 0.15)