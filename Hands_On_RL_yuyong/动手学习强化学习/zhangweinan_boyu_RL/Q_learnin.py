import time 
import random


class Env():
    def __init__(self, length, height):
        # define the height and length of the map
        self.length = length
        self.height = height
        # define the agent's start position
        self.x = 0
        self.y = 0

    def render(self, frames=50):
        for i in range(self.height):
            if i == 0: # cliff is in the line 0
                line = ['S'] + ['x']*(self.length - 2) + ['T'] # 'S':start, 'T':terminal, 'x':the cliff
            else:
                line = ['.'] * self.length
            if self.x == i:
                line[self.y] = 'o' # mark the agent's position as 'o'
        #     print(''.join(line))
        # print('\033['+str(self.height+1)+'A')  # printer go back to top-left 
        # time.sleep(1.0 / frames)

    def step(self, action):
        """4 legal actions, 0:up, 1:down, 2:left, 3:right"""
        change = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        self.x = min(self.height - 1, max(0, self.x + change[action][0]))
        self.y = min(self.length - 1, max(0, self.y + change[action][1]))

        states = [self.x, self.y]
        reward = -1
        terminal = False
        if self.x == 0: # if agent is on the cliff line "SxxxxxT"
            if self.y > 0: # if agent is not on the start position 
                terminal = True
                if self.y != self.length - 1: # if agent falls
                    reward = -100
        return reward, states, terminal

    def reset(self):
        self.x = 0
        self.y = 0

class Q_table():
    def __init__(self, length, height, actions = 4, alpha = 0.1, gamma = 0.9): 
        self.table = [0] * (length * height * actions) # initialize the Q-table with 0
        self.length = length
        self.height = height
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma


    def _index(self,a,x,y):
        return a * self.length * self.height + x * self.length + y
    
    def _epsilon(self, num_episode):
        # return 0.1
        # version for better convergence:
        # """At the beginning epsilon is 0.2, after 300 episodes decades to 0.05, and eventually go to 0."""
        return 20. / (num_episode + 100)

    def take_action(self, x, y,   num_episode):
        if random.random() < self._epsilon(num_episode): # epsilon-greedy action selection
            return random.randint(0, self.actions - 1)
        else:
            actions_value = [self.table[self._index(a,x,y)] for a in range(self.actions)]
            return actions_value.index(max(actions_value))
    def max_q(self, x, y):
        return max([self.table[self._index(a,x,y)] for a in range(self.actions)])
    
    def update(self, a,s0,s1,r,is_terminal):
        q_predict = self.table[self._index(a,s0[0],s0[1])]
        if is_terminal:
            q_target = r
        else:
            q_target = r + self.gamma * self.max_q(s1[0], s1[1])
        self.table[self._index(a,s0[0],s0[1])] += self.alpha * (q_target - q_predict)



env = Env(12, 4)
q_table = Q_table(12, 4)
for episode in range(5000):
    episode_reward = 0
    s0 = [0, 0]
    is_terminal = False
    while not is_terminal:
        env.render()
        action = q_table.take_action(s0[0], s0[1], episode)
        r,s1,is_terminal = env.step(action)
        q_table.update(action, s0,s1, r,is_terminal)
        episode_reward += r
        s0 = s1
    if episode % 100 == 0:
        print('Episode: {}, episode reward: {}'.format(episode, episode_reward))
    env.reset() 