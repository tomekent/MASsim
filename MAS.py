#!/usr/bin/python3.5

import os
import logging
import random
from datetime import datetime
import copy
import numpy as np
import time
import dill


from tools import visualisation, postprocessing
from tools.states import worldState


class MAS():
    def __init__(self, noAgents, plot=False):
        self.basename = 'MAS'
        self.noAgents = noAgents
        self.Routes = [[]]*noAgents
        self.store_results = False
        self.settings = {'plot_updates': plot, 'save_setup': False, 'useRadius': True, 'useComms': True, 'commsRadius': 50.0, 'considerationRadius': 10.,
                             'blackout': False, 'hopNo': 3}

        self.timers = {'demeIter': [0.0, 0.0], 'exchange': [0.0, 0.0], 'visualisation': [0.0, 0.0],
                       'completingTasks': [0.0, 0.0]}

        self.datestr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.foldername = os.path.join(os.getcwd(), self.basename, 'results', self.datestr)

        self.setup_data()
        self.dir_to_vec = {'North': [0,1], 'South': [0,-1], 'East': [1,0], 'West': [-1,0]}

        self.speeds = [5.0] * self.noAgents
        self.allResults = []

        self.iteration = 0

        if self.settings['save_setup'] or self.settings['plot_updates']:
            self.setup_folders()
            if self.settings['save_setup']:
                self.save_setup()


        loiterScheme = 'None'
        self.setup_agents(self.settings['commsRadius'], loiterScheme)



    def setup_folders(self):
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)


    def setup_data(self):
        # intialise randomly
        self.state = worldState(samp_dist=100, samp_radius=5)
        self.state.spawn_agents(self.noAgents)


    def save_setup(self):
        # lets save the state to a pkl file for later
        filename = self.foldername + '/setup_data.pkl'

        pickle_data = {'noAgents': self.noAgents, 'state': self.state}
        with open(filename, 'wb') as f:
            # for k, v in pickle_data.iteritems():
            dill.dump(pickle_data, f)

    def setup_agents(self, commsRadius, loiterScheme):

        for a in self.state.agentRange:
            self.state.agents[a].commsRadius = commsRadius
            self.state.agents[a].v = self.speeds[a]
            self.state.agents[a].loiterScheme = loiterScheme



    def plot_initial_state(self):
        iter = '_i%03d' % self.iteration
        visualisation.plot_result(self.state, self.Routes, logbook=None, foldername=self.foldername, save_plot=True, suffix=iter, commsRadius=self.settings['useComms'],considerationRadius=self.settings['considerationRadius'])



    # # run the iterations
    # while True:
    # # for j in range(0,5):
    def degrade_comms(self, type='None'):
        if type == 'Random':
            degradeFactor = 0.1
            degradeProb = 0.05
            # random degradation of comms?
            for agent in self.state.agents:
                if random.random() < degradeProb:
                    agent.degradeComms(degradeFactor)
                elif random.random() < degradeProb:
                    agent.degradeComms(-degradeFactor*0.5)
        else:
            if self.state.simTime > 5.0:
                if self.state.simTime < 25.0:
                    for agent in self.state.agents:
                        agent.degradeComms(0.25)
                elif self.state.simTime < 35.0:
                    for agent in self.state.agents:
                        agent.improveComms(0.25)


    def iterate(self, actions=None):
        self.iteration += 1
        logging.info('Running Iteration %i' % self.iteration)
        if actions == None:
            pass
        else:
            if len(actions) != self.noAgents:
                actions = [actions[0] for _ in range(self.noAgents)]  # just copy the first one to all
            a = 0
            for action in actions:
                if type(action) is str:
                    self.state.agents[a].move_direction(self.dir_to_vec[action], self.state.agents[a].v, self.state.dt)
                    a += 1
                elif type(action) is np.ndarray or type(action) is float:
                    if type(action) is np.ndarray:
                        action = action[0]
                    vec = [np.cos(action), np.sin(action)]
                    self.state.agents[a].move_direction(vec, self.state.agents[a].v, self.state.dt)
                    a += 1

        # lets make sure they don't leave
        for agent in self.state.agents:
            agent.current_location[0] = max(self.state.xlimits[0], agent.current_location[0])
            agent.current_location[0] = min(self.state.xlimits[1], agent.current_location[0])
            agent.current_location[1] = max(self.state.ylimits[0], agent.current_location[1])
            agent.current_location[1] = min(self.state.ylimits[1], agent.current_location[1])

        ## degrade comms?
        if self.settings['blackout']:
           self.degrade_comms()

        if self.settings['plot_updates']:
            self.plot_iteration()

        self.update_state()
        # t5 = time.time() - t0


        if self.store_results:
            print("Storing results")
            Results = {}
            Results['Iteration'] = copy.deepcopy(self.iteration)
            Results['simTime'] = self.state.simTime
            Results['DistanceTravelled'] = copy.deepcopy([agent.distanceTravelled for agent in self.state.agents])
            Results['AgentLocations'] = copy.deepcopy([agent.current_location for agent in self.state.agents])
            Results['timers'] = self.timers
            self.allResults.append(Results)



    def plot_iteration(self, save_plot=True):
        ############ Plot ############
        iter = '_i%03d' % self.iteration
        self.timers['visualisation'][0] = time.time()
        if not hasattr(self, 'Routes'):
            self.plot_initial_state()
        else:
            ax, lines = visualisation.plot_result(self.state, self.Routes, [], logbook=None, foldername=self.foldername, save_plot=save_plot, suffix=iter, commsRadius=self.settings['useComms'], considerationRadius=self.settings['considerationRadius'])
        self.timers['visualisation'][1] = time.time()



    def update_state(self):
        ############ Update the sim states ############
        dt = 1.0
        # v = 5.0
        self.state.simTime += dt # increment the timer in simTime
        self.state.update()


    def check_complete(self):
        return False



    def save_final_results(self):
        ############ Plot Post Processing Results ############
        postprocessing.plot_global_progress(self.allResults, self.state, self.foldername)

        # lets save the results
        filename = self.foldername + '/results_data_MAS_' + self.datestr + '.pkl'

        pickle_data = {'noAgents': self.noAgents, 'state': self.state, 'results': self.allResults}
        with open(filename, 'wb') as f:
            # for k, v in pickle_data.iteritems():
            dill.dump(pickle_data, f)


if __name__ == "__main__":

    noAgents = 10
    mas_sim = MAS(noAgents)
    mas_sim.setup_folders() #create folders for saving plots

    action_strs = ['North', 'South', 'East', 'West']

    # lets try some discrete actions
    for i in range(1, 10):
        actions = [random.choice(action_strs) for _ in range(noAgents)]#direction
        mas_sim.iterate(actions)
        mas_sim.plot_iteration()

    # lets try some continous actions
    for i in range(1, 10):
        actions = [random.uniform(-np.pi, np.pi) for _ in range(noAgents)] #angle in radians
        mas_sim.iterate(actions)
        mas_sim.plot_iteration()
