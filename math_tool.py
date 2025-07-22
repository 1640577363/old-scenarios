import numpy as np
import math

# 1 simulate lidar
def update_lasers(pos, obs_pos, r, L, num_lasers, bound):

    distance_to_obs = np.linalg.norm(np.array(pos) - np.array(obs_pos))
    isInObs = distance_to_obs < r \
                or pos[0] < 0 \
                or pos[0] > bound \
                or pos[1] < 0 \
                or pos[1] > bound
    
    if isInObs:
        #如果在障碍物内或超出边界，返回所有激光距离为0，并设置标志
        return [0.0] * num_lasers, isInObs

    #生成等间隔的激光角度（0到2π）
    # endpoint=False 确保不包含2π（即0°和360°不重复）
    angles = np.linspace(0, 2 * np.pi, num_lasers, endpoint=False)
    #初始化所有激光距离为最大探测距离L
    laser_lengths = [L] * num_lasers

    '''
        遍历每个激光角度：
    
        计算与障碍物的交点距离
        
        如果交点距离小于当前激光长度，更新激光距离'''
    for i, angle in enumerate(angles):
        intersection_dist = check_obs_intersection(pos, angle, obs_pos, r, L)
        if laser_lengths[i] > intersection_dist:
            laser_lengths[i] = intersection_dist

    '''再次遍历每个激光角度：

        计算与边界的交点距离
        
        如果边界距离小于当前激光长度（且小于障碍物距离），更新激光距离
    '''
    for i, angle in enumerate(angles):
        wall_dist = check_wall_intersection(pos, angle, bound, L)
        if laser_lengths[i] > wall_dist:
            laser_lengths[i] = wall_dist
    #返回更新后的激光距离列表和障碍物标志
    return laser_lengths, isInObs

# 障碍物交点检测函数
def check_obs_intersection(start_pos, angle, obs_pos,r,max_distance):
    ox = obs_pos[0]
    oy = obs_pos[1]
    #计算激光终点坐标（假设无障碍物）
    end_x = start_pos[0] + max_distance * np.cos(angle)
    end_y = start_pos[1] + max_distance * np.sin(angle)

    #计算激光方向的向量分量
    dx = end_x - start_pos[0]
    dy = end_y - start_pos[1]
    fx = start_pos[0] - ox
    fy = start_pos[1] - oy

    #计算二次方程的系数
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = (fx**2 + fy**2) - r**2
    discriminant = b**2 - 4 * a * c#计算判别式，确定交点数量

    '''如果有实根（激光与圆相交）：

    计算判别式的平方根
    
    计算两个可能的交点参数t1和t2'''
    if discriminant >= 0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        if 0 <= t1 <= 1:
            return t1 * max_distance
        if 0 <= t2 <= 1:
            return t2 * max_distance
    return max_distance

#边界交点检测函数
def check_wall_intersection(start_pos, angle, bound, L):
    #计算方向余弦和正弦

# 初始化交点距离为最大长度L
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    L_ = L

    #  (y = bound)
    if sin_theta > 0:  
        L_ = min(L_, abs((bound - start_pos[1]) / sin_theta))

    #  (y = 0)
    if sin_theta < 0:  
        L_ = min(L_, abs(start_pos[1] / -sin_theta))

    #  (x = bound)
    if cos_theta > 0: 
        L_ = min(L_, abs((bound - start_pos[0]) / cos_theta))
    
    #  (x = 0)
    if cos_theta < 0: 
        L_ = min(L_, abs(start_pos[0] / -cos_theta))

    return L_

def cal_triangle_S(p1, p2, p3):
    S = abs(0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])))
    if math.isclose(S, 0.0, abs_tol=1e-9):
        return 0.0
    else:
        return S