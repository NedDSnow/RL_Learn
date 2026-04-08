import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()
#   这里定义的P是一个三维列表,第一维是状态,第二维是动作,第三维是一个列表,包含了所有可能的转移结果(概率、下一个状态、奖励、是否结束)。在这个环境中,每个状态和动作对应的转移结果只有一个,所以第三维的列表中只有一个元素。
# 这里的状态是按照行优先的顺序编号的,例如在一个4行12列的网格世界中,第一行的状态编号为0-11,第二行的状态编号为12-23,以此类推。动作的编号为0-3,分别对应上、下、左、右四个方向。转移结果中的奖励和是否结束是根据环境的规则定义的,例如在悬崖漫步环境中,如果下一个状态是悬崖或者目标状态,奖励为-100或者0,并且结束标志为True;否则奖励为-1,结束标志为False。
# P中的 的构建 非常巧妙,通过循环遍历每个状态和动作,根据环境的规则计算下一个状态、奖励和结束标志,并将这些信息存储在P中。这样就构建了一个完整的状态转移矩阵,可以用于后续的策略评估和策略提升等算法。
    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置，根据动作计算下一个位置,并根据环境规则计算奖励和结束标志
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点， 构建比较重要
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
# 上述代码定义了一个悬崖漫步环境,其中createP函数构建了状态转移矩阵P,描述了在每个状态下执行每个动作后可能的结果(下一个状态、奖励、是否结束)。这个环境可以用于测试和比较不同的强化学习算法,例如策略迭代和价值迭代等。
    
class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.pi = [[0.25, 0.25, 0.25, 0.25]
                   for i in range(self.env.ncol * self.env.nrow)]  # 初始化为均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):  # 策略评估
        cnt = 1  # 计数器
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                        # 本章环境比较特殊,奖励和下一个状态有关,所以需要和状态转移概率相乘
                    qsa_list.append(self.pi[s][a] * qsa) ## 计算状态s下执行动作a的Q(s,a)价值,并乘以当前策略pi[s][a]的概率,得到状态s的价值函数V(s)的一部分。因为状态s的价值函数V(s)是所有动作a的Q(s,a)价值的加权和,权重就是当前策略pi[s][a]的概率。
                new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系 、针对四个动作的Q(s,a)价值进行加权求和,得到状态s的价值
                max_diff = max(max_diff, abs(new_v[s] - self.v[s])) # 计算状态s的价值更新前后差值的绝对值,并更新最大差值
            self.v = new_v # 更新状态价值函数
            if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代
            cnt += 1 # 评估迭代次数加1
        print("策略评估进行%d轮后完成" % cnt)

    def policy_improvement(self):  # 策略提升
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done)) ## 计算状态s下执行动作a的Q(s,a)价值,与策略评估中计算状态s的价值的方式类似,但是这里没有乘以策略概率,因为我们需要比较不同动作的Q(s,a)价值,从而选择最优动作。
                qsa_list.append(qsa)
            maxq = max(qsa_list) ## 找到状态s下的最大Q(s,a)价值,这个最大值对应的动作就是当前策略下状态s的最优动作。
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):  # 策略迭代
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,方便接下来进行比较
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break
            
def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


# env = CliffWalkingEnv()
# action_meaning = ['^', 'v', '<', '>']
# theta = 0.001
# gamma = 0.9
# agent = PolicyIteration(env, theta, gamma)
# agent.policy_iteration()
# print_agent(agent, action_meaning, list(range(37, 47)), [47])



class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.theta = theta  # 价值收敛阈值
        self.gamma = gamma
        # 价值迭代结束后得到的策略
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)  # 这一行和下一行代码是价值迭代和策略迭代的主要区别
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()

    def get_policy(self):  # 根据价值函数导出一个贪婪策略
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]


# env = CliffWalkingEnv()
# action_meaning = ['^', 'v', '<', '>']
# theta = 0.001
# gamma = 0.9
# agent = ValueIteration(env, theta, gamma)
# agent.value_iteration()
# print_agent(agent, action_meaning, list(range(37, 47)), [47])



import gymnasium as gym
env = gym.make("FrozenLake-v1")  # 创建环境
env = env.unwrapped  # 解封装才能访问状态转移矩阵P
env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)

for a in env.P[14]:  # 查看目标左边一格的状态转移信息
    print(env.P[14][a])


# 这个动作意义是Gym库针对冰湖环境事先规定好的
# 策略迭代算法的一个优点是,它能够保证在每次策略提升后得到一个更优的策略,从而最终收敛到最优策略。通过交替进行策略评估和策略提升,策略迭代算法能够逐步改进策略,直到无法再改进为止。这种机制使得策略迭代算法能够在有限的时间内找到最优策略,并且在某些环境中可能会比价值迭代算法更快地收敛。然而,策略迭代算法也有一些缺点,例如在某些环境中可能需要较多的评估步骤才能收敛,从而导致计算成本较高。因此,在实际应用中,我们需要根据具体的环境和问题特点来选择合适的算法。
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])


# 价值迭代算法的一个优点是,它不需要像策略迭代算法那样进行策略评估,因此在某些情况下可能会更快地收敛到最优策略。价值迭代算法通过直接更新状态价值函数来寻找最优策略,而不需要显式地计算和评估策略。这使得价值迭代算法在某些环境中可能会更高效,尤其是在状态空间较大或者策略评估较为复杂的情况下。然而,价值迭代算法也有一些缺点,例如在某些环境中可能会出现震荡现象,导致收敛速度变慢。因此,在实际应用中,我们需要根据具体的环境和问题特点来选择合适的算法。
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])
