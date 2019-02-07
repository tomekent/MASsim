import gym
from gym import error, spaces, utils
from tools.hexes import  Point
import os
import logging
import random
import numpy as np
import time

from MAS.MAS import MAS


class MAS_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, noAgents, type='Discrete', action_limit=10, max_stuck=15, loadfile=False):
        self.status = 'STARTING'
        self.type = type
        self.max_stuck = max_stuck
        # noAgents = 5
        self.noAgents = noAgents
        self.plot = False
        self.loadfile = loadfile
        self.action_limit = action_limit
        self.action_count = 0
        self.create_folder = True
        self.mas = MAS(self.noAgents, self.plot)
        self.mas.store_results = False
        self.max_delay = 10
        self.idle_score = self.mas.state.hex_score_max * self.noAgents
        self.last_score = 0 #self.mas.state.hex_score_max * self.noAgents
        self.last_alive = 1 * self.noAgents
        self.best_score = 0
        self.steps_since_improve = 0
        self.hexkeys = [hexkey for hexkey in self.mas.state.hex_grid.hexes.keys()]
        self.out_of_bounds = False

        low = np.array([-1])  # , 0])
        high = np.array([1])  # , self.max_delay])
        # noActions =  len(self.hex_set_i) # len(self.hexkeys)


        if self.type == 'Discrete':
            self.actions = ['North', 'South', 'East', 'West']
            noActions =  len(self.actions)
            self.action_space = spaces.Discrete(noActions)
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.mas.state.hex_grid.n_max, self.mas.state.hex_grid.m_max, 2),
                                                dtype=np.uint8)
        else:
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            noHexes = len(self.hexkeys)
            obs_size = 7
            hexVals_high = np.array([50.0] * obs_size)
            hexVals_low = np.array([0.] * obs_size)

            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self.observation_space = spaces.Box(low=hexVals_low, high=hexVals_high, dtype=np.float32)


    def step(self, action):
        self.status = 'RUNNING'
        self.take_action(action)

        reward = self.get_reward()
        if self.type == 'Discrete':
            ob = self.get_state()
        else:
            ob = self.get_state_hex_edges()

        episode_over = (self.status == 'FINISHED')
        # if episode_over:
        #     self.mas.print_timers()
        return ob, reward, episode_over, {}

    def get_state_grid(self):

        N = len(self.mas.state.hex_grid.hexes.items())
        rows = []
        cols = []
        for hexkey, hexi in self.mas.state.hex_grid.hexes.items():
            rows.append(hexi.row)
            cols.append(hexi.col)

        row_off = -min(rows)
        col_off = -min(cols)
        print(row_off, col_off)
        n_max = max(rows) + row_off
        m_max = max(cols) + col_off

        import numpy as np
        mat = np.zeros((n_max + 1, m_max + 1), dtype=np.int32)
        print(n_max, m_max)
        for i in range(N):
            mat[rows[i] + row_off][cols[i] + col_off] = cols[i]

    def get_state_hex_edges(self):
        agent = self.mas.state.agents[0]
        hex = self.mas.state.hex_grid.layout.pixel_to_hex(Point(agent.current_location[0], agent.current_location[1]))
        hex_adj = self.mas.state.hex_grid.layout.hex_adjacent(hex)
        hexes = [hex] + hex_adj
        obs = []
        for hex in hexes:
            hex_key = (hex.q, hex.r)
            if hex_key in self.mas.state.hex_grid.hexinfo:
                obs.append(self.mas.state.hex_grid.hexinfo[hex_key]['score'])
            else:
                obs.append(50)
        # print(obs)
        if self.action_count > self.action_limit:
            self.status = 'FINISHED'
        if self.steps_since_improve > self.max_stuck:
            self.status = 'FINISHED'
        return obs

    def get_state(self):
        scoreMax = 20.
        agentsMax = 3
        scores = [self.mas.state.hex_grid.hexinfo[key]['score'] for key in self.hexkeys]
        agentCounts = [self.mas.state.hex_grid.hexinfo[key]['AgentCount'] for key in self.hexkeys]
        agent_locs_x = [agent.current_location[0] for agent in self.mas.state.agents]
        agent_locs_y = [agent.current_location[1] for agent in self.mas.state.agents]

        if self.type == 'Discrete':
            obs = np.zeros((self.mas.state.hex_grid.n_max, self.mas.state.hex_grid.m_max, 2), dtype='uint8')
            for hexkey, hexi in self.mas.state.hex_grid.hexes.items():
                row = hexkey[0]
                col = hexkey[1]
                obs[row + self.mas.state.hex_grid.row_off][col + self.mas.state.hex_grid.col_off][0] = int(
                    255 * min((self.mas.state.hex_grid.hexinfo[hexkey]['score'] / scoreMax), 1))

                obs[row + self.mas.state.hex_grid.row_off][col + self.mas.state.hex_grid.col_off][1] = int(
                    255 * min((self.mas.state.hex_grid.hexinfo[hexkey]['AgentCount'] / agentsMax), 1))
        else:
            obs = scores + agent_locs_x + agent_locs_y

        if self.action_count > self.action_limit:
            self.status = 'FINISHED'
        if self.steps_since_improve > self.max_stuck:
            self.status = 'FINISHED'

        return obs

    def reset(self):
        try:
            self.mas = MAS(self.noAgents, self.plot)
        except:
            self.mas = MAS(self.noAgents, self.plot)

        self.mas.store_results = False
        self.action_count = 0
        # self.last_score = 0  # self.mas.state.hex_score_max * self.noAgents
        # self.last_alive = 1 * self.noAgents
        self.create_folder = True
        self.mas.iterate()
        if self.type == 'Discrete':
            obs = self.get_state()
        else:
            obs = self.get_state_hex_edges()

        scores = [self.mas.state.hex_grid.hexinfo[key]['score'] for key in self.hexkeys]
        self.last_alive = sum([int(score > 0) for score in scores])
        self.last_score = sum(scores)
        self.best_score = sum(scores)
        self.steps_since_improve = 0

        return obs

    def render(self, mode='human', close=False):
        if self.create_folder:
            self.mas.setup_folders()
            self.create_folder = False
        self.mas.plot_iteration()
        pass

    def take_action(self, action):
        if self.type == 'Discrete':
            self.mas.iterate(self.actions[action])
        else:
            for i in range(0, 3):
                self.mas.iterate(np.pi*action) #move in the direction of pi *action
                # lets make sure they don't leave
            self.out_of_bounds = False
            for agent in self.mas.state.agents:
                if agent.current_location[0] < self.mas.state.xlimits[0]:
                    agent.current_location[0] = self.mas.state.xlimits[0]
                    self.out_of_bounds = True
                if agent.current_location[0] > self.mas.state.xlimits[1]:
                    agent.current_location[0] = self.mas.state.xlimits[1]
                    self.out_of_bounds = True
                if agent.current_location[1] < self.mas.state.ylimits[0]:
                    agent.current_location[1] = self.mas.state.ylimits[0]
                    self.out_of_bounds = True
                if agent.current_location[1] > self.mas.state.ylimits[1]:
                    agent.current_location[1] = self.mas.state.ylimits[1]
                    self.out_of_bounds = True

        self.action_count += 1

        pass

    def get_reward(self):
        scores = [self.mas.state.hex_grid.hexinfo[key]['score'] for key in self.hexkeys]
        total_alive = sum([int(score > 0.25) for score in scores])
        total_score = sum(scores)
        # reward = total_score - self.last_score
        reward = 0 # total_alive - self.last_alive
        if total_score > self.best_score:
            self.steps_since_improve = 0
            reward = max(total_score - self.best_score, 0)
        else:
            self.steps_since_improve += 1
        # if total_score < self.last_score:
        # #     reward = 1
        # # # elif total_alive >= self.last_alive:
        # # #     reward = 0
        # # else:
        #     reward = -1
        # if total_alive > self.last_alive:
        #     reward += 1

        # reward = self.last_alive - total_alive

        if self.out_of_bounds:
            reward = -50
        self.last_score = total_score
        self.last_alive = total_alive
        self.best_score = max(self.best_score, total_score)
        # """ Reward is given for XY. """
        # print("reward given : %2.2f" % reward)
        return reward