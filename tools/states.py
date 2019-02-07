from tools.assignment_tools import distance, generate_samples
from tools.agents import Agent, eaAgent
from tools.tasks import Task, Tasker, randomTasker, blobTasker, hexTasker
from tools.hexes import hexGrid, Point
import numpy as np
import random

class worldState():
    def __init__(self, samp_dist = 200, samp_radius = 5):
        self._agents = []
        self._tasks = []
        self._noAgents = 0
        self._noTasks = 0
        self.simTime = 0.0
        self.dt = 1.0
        self.HL = 120.0
        self.hex_score_max = 20.0
        self.hex_size = 10

        self.locations = generate_samples(samp_dist, samp_radius, 10)

        random.shuffle(self.locations)
        # print len(self.locations)
        # self.task_locations = self.locations[:int((len(self.locations)/2))]
        self.agent_locations = self.locations[int((len(self.locations)/2)):]

        self.xlimits = [0.0, samp_dist]
        self.ylimits = [0.0, samp_dist]
        self.home_location = [(self.xlimits[1] - self.xlimits[0]) /2., (self.ylimits[1] - self.ylimits[0]) /2.]
        # self.hex_grid = hexGrid(Point(self.xlimits[0]-5.0,self.ylimits[0]-5.0), Point(20, 20), Point(self.xlimits[1]+10.0, self.ylimits[1]+10.0)
        self.hex_grid = hexGrid(Point(self.xlimits[0]-5.0,self.ylimits[0]-5.0), Point(self.hex_size, self.hex_size), Point(self.xlimits[1]+0.0, self.ylimits[1]+0.0))
        self.tasker = hexTasker(self.hex_grid)  # blobTasker(noBlobs=5)

    @property
    def tasks(self):
        return self._tasks

    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, list):
        self._agents = list
        self._noAgents = self.noAgents
        print(self._agents)
        print(self._noAgents)

    @tasks.setter
    def tasks(self, list):
        self._tasks = list
        self._noTasks = self.noTasks
        print(self._tasks)
        print(self._noTasks)

    @property
    def noAgents(self):
        return len(self.agents)

    @property
    def noTasks(self):
        return len(self.tasks)

    @property
    def agentRange(self):
        return range(0, self.noAgents)

    @property
    def taskRange(self):
        return range(0, self.noTasks)

    def spawn_agents(self, noAgents):
        for i in range(0, noAgents):
            newAgent = Agent()
            # print i,
            # print self.agent_locations[i]
            newAgent.set_start(self.agent_locations[i])
            # newAgent.weight = (i+1) *0.5
            newAgent.no = i
            self._agents.append(newAgent)

    def spawn_new_agents(self, noNew):
        noAgents = self.noAgents
        for i in range(0, noNew):
            newAgent = Agent()
            newAgent.set_start(self.agent_locations[noAgents + i])
            newAgent.no = noAgents + i
            self._agents.append(newAgent)

    def spawn_defined_agents(self, locations):
        start_i = self.noAgents
        for i in range(0, len(locations)):
            newAgent = Agent()
            # print i,
            # print self.agent_locations[i]
            newAgent.set_start(locations[i])
            # newAgent.weight = (i+1) *0.5
            newAgent.no = start_i + i
            self._agents.append(newAgent)

    def spawn_tasks(self, num):
        for i in range(0, num):
            newTask = self.tasker.spawn_task(i)
            # newTask = Task()
            # newTask.set_location(self.task_locations[i])
            # newTask.complete = False
            # newTask.no = i
            newTask.spawnedAt = self.simTime
            self._tasks.append(newTask)

    def spawn_new_tasks(self, noNew, types=None):
        noTasks = self.noTasks
        newTaskNos = []
        for i in range(0, noNew):
            newTask = self.tasker.spawn_task(noTasks + i)
            # # newTask = Task()
            # newTask.set_location(self.task_locations[noTasks + i])
            # newTask.complete = False
            # newTask.no = noTasks + i
            newTask.spawnedAt = self.simTime
            if types is not None:
                if len(types) == noNew:
                    newTask.type = types[i]
                else:
                    newTask.type = types
            self._tasks.append(newTask)
            newTaskNos.append(newTask.no)
        return newTaskNos


    def spawn_defined_tasks(self, locations):
        newTaskNos = []
        start_i = self.noTasks
        for i in range(0, len(locations)):
            newTask = Task()
            newTask.set_location(locations[i])
            newTask.complete = False
            newTask.no = start_i + i
            self._tasks.append(newTask)
            newTaskNos.append(newTask.no)
        return newTaskNos


    def generate_cost_matrix(self):
        C = []
        for i in range(0, self.noAgents):
            C.append([0.] * self.noTasks)
        for a, agent in enumerate(self._agents):
            for t, task in enumerate(self._tasks):
                C[a][t] = agent.distance_to_target(task.location) * agent.weight
            # 	print a, t,
            # 	print C[a][t]
            # print C

        return C

    def generate_agent_to_task_distance_matrix(self):
        D = np.zeros([self.noAgents, self.noTasks])
        # for i in range(0, self.noAgents):
        #     D.append([0.] * self.noTasks)
        for a, agent in enumerate(self._agents):
            for t, task in enumerate(self._tasks):
                D[a][t] = agent.distance_to_target(task.location)

        return D

    def generate_task_distance_matrix(self, oldD=None):
        # we can assume tasks don't move so no need to recalculate?

        if oldD is None:
            D = np.zeros([self.noTasks, self.noTasks])
            # for i in range(0, self.noTasks):
            #     D.append([0.] * self.noTasks)
            #
            for t1, task1 in enumerate(self._tasks):
                for t2, task2 in enumerate(self._tasks):
                    D[t1][t2] = distance(task1.location, task2.location)
                # 	print a, t,
                # 	print D[a][t]
                # print D
        else:
            D = oldD[self.noAgents:, self.noAgents:]
            nt = len(D)  # no tasks already in there
            if self.noTasks == nt:
                return D

            nadd = self.noTasks - nt
            if nt == 0:
                newD = []
                for i in range(nt, self.noTasks):  # add new rows
                    newD.append([distance(self._tasks[i].location, self._tasks[t2].location) for t2 in range(0, self.noTasks)])
                D = np.array(newD)
            else:
                cols = []
                for i in range(0, nt): # extend the cols
                    cols.append([distance(self._tasks[i].location, self._tasks[t2].location) for t2 in range(nt, self.noTasks)])
                D = np.concatenate((D, np.array(cols)), axis = 1)
                    # D[i] = np.hstack((D[i], np.array([distance(self._tasks[i].location, self._tasks[t2].location) for t2 in range(nt, self.noTasks)])))
                rows = []
                for i in range(nt, self.noTasks): # add new rows
                    rows.append([distance(self._tasks[i].location, self._tasks[t2].location) for t2 in range(0, self.noTasks)])
                D = np.concatenate((D, np.array(rows)), axis=0)




        return D

    def generate_full_distance_matrix(self, oldD = None):
        dA = self.generate_agent_to_task_distance_matrix()
        dT = self.generate_task_distance_matrix(oldD = oldD)

        D = np.concatenate((dA, dT), axis=0)
        left = np.zeros((self.noTasks + self.noAgents, self.noAgents))
        D = np.concatenate((left, D), axis=1)


        # print(len(D), len(D[0]))
        return D

    def update_hexes(self):
        for hexkey in self.hex_grid.hexes.keys():
            self.hex_grid.hexinfo[hexkey]['occupied'] = False
            self.hex_grid.hexinfo[hexkey]['AgentCount'] = 0
            self.hex_grid.hexinfo[hexkey]['TaskCount'] = 0
            self.hex_grid.hexinfo[hexkey]['score'] *= (1./2) ** (self.dt / self.HL) #0.975 # degrade the score

        for t in self.taskRange:
            if not self.tasks[t].complete:
                loc = self.tasks[t].location
                hex = self.hex_grid.layout.pixel_to_hex(Point(loc[0], loc[1]))
                self.hex_grid.hexinfo[(hex.q, hex.r)]['TaskCount'] += 1

        for a in self.agentRange:
            loc = self.agents[a].current_location
            hex = self.hex_grid.layout.pixel_to_hex(Point(loc[0], loc[1]))
            self.hex_grid.hexinfo[(hex.q, hex.r)]['AgentCount'] += 1
            if self.hex_grid.hexinfo[(hex.q, hex.r)]['occupied']:
                pass # already occupied by another agent
            else:
                self.hex_grid.hexinfo[(hex.q, hex.r)]['occupied'] = True
                self.hex_grid.hexinfo[(hex.q, hex.r)]['last_visited'] = self.simTime
                self.hex_grid.hexinfo[(hex.q, hex.r)]['score'] = min(self.hex_grid.hexinfo[(hex.q, hex.r)]['score'] + (self.dt * 5), self.hex_score_max)



    def move_agents(self):
        for a in self.agentRange:
            a_route = self.agents[a].currentRoute
            if len(a_route) > 0:
                next_task = a_route[0]
                next_task_location = self.tasks[next_task].location
                self.agents[a].move_toward(next_task_location, self.agents[a].v, self.dt)
            else: # should we do some loitering?
                next_location = self.loiter_to(a, scheme=self.agents[a].loiterScheme)
                if next_location is not None:
                    self.agents[a].move_toward(next_location, self.agents[a].v*0.5, self.dt)


    def loiter_to(self, agentNo, scheme='Random'):
        if scheme == 'Random':
            # we will either go North East South West
            direction = random.choice(['North', 'South', 'East', 'West'])
            if direction == 'North':
                next_location = [self.agents[agentNo].current_location[0] + 0., min(self.agents[agentNo].current_location[0] + 20, self.ylimits[1])]
            elif direction == 'South':
                next_location = [self.agents[agentNo].current_location[0] + 0., max(self.agents[agentNo].current_location[0] - 20, self.ylimits[0])]
            elif direction == 'East':
                next_location = [min(self.agents[agentNo].current_location[0] + 20., self.xlimits[1]), self.agents[agentNo].current_location[0] + 0]
            else:
                next_location = [max(self.agents[agentNo].current_location[0] - 20., self.xlimits[0]), self.agents[agentNo].current_location[0] + 0]
        elif scheme == 'NearestAgent':
            distanceToAgents = [self.agents[a].distance_to_target(self.agents[a].current_location) for a in self.agentRange]
            distanceToAgents[agentNo] = 1e9
            mina = distanceToAgents.index(min(distanceToAgents))
            next_location = self.agents[mina].current_location
        elif scheme == 'Flock':
            delta_c = 10.0
            xys = np.array([agent.current_location for agent in self.agents])
            loc = self.agents[agentNo].current_location
            # # def separation(self, i, near, s):
            # dx_s = np.mean(loc - xys, axis=0)
            # dx_s = dx_s / np.linalg.norm(dx_s)
            # def cohesion(self, i, near, s):
            mx = np.mean(xys, axis=0)
            # dx_c = mx - loc
            # dx_c = dx_c / np.linalg.norm(dx_c)
            next_location = mx# [loc[0] + dx_c[0]*delta_c, loc[1] + dx_c[1]*delta_c]
        elif scheme == 'FlockTasks':
            delta_c = 10.0
            xys = np.array([task.location for task in self.tasks if not task.complete])
            if len(xys) == 0:
                next_location = self.agents[agentNo].current_location
            else:
                # loc = self.agents[agentNo].current_location
                # # def separation(self, i, near, s):
                # dx_s = np.mean(loc - xys, axis=0)
                # dx_s = dx_s / np.linalg.norm(dx_s)
                # def cohesion(self, i, near, s):
                mx = np.mean(xys, axis=0)
                # dx_c = mx - loc
                # dx_c = dx_c / np.linalg.norm(dx_c)
                next_location = mx# [loc[0] + dx_c[0]*delta_c, loc[1] + dx_c[1]*delta_c]
        elif scheme == 'Home':
            next_location = self.home_location
        else:
            return None
        return next_location

    def update_task_completion(self):
        radius = 1.0
        for ti in range(self.noTasks):
            if not self._tasks[ti].complete:
                for agent in self._agents:
                    if ti in agent.currentRoute: # make sure its only completed by the assigned agent
                        if agent.distance_to_target(self._tasks[ti].location) < radius:
                            self._tasks[ti].completed(agent.no, self.simTime)
                            # print('Task %i complete by agent %i' %(ti, agent.no))



    def all_tasks_complete(self):
        completeCount = 0
        for ti in range(self.noTasks):
            if self.tasks[ti].complete:
                completeCount += 1

        # print(self.noTasks, completeCount)
        return completeCount == self.noTasks

    def update(self):
        for agent in self.agents:
            agent.locationHistory.append([self.simTime, agent.current_location[0], agent.current_location[1]])
        self.move_agents()
        self.update_task_completion()
        self.update_hexes()


class worldStateEA(worldState):
    def __init__(self, samp_dist = 100, samp_radius = 5):
        super().__init__(samp_dist = samp_dist, samp_radius = samp_radius)

    def spawn_agents(self, noAgents, type=None, weights=None):
        for i in range(0, noAgents):
            newAgent = eaAgent()
            newAgent.set_start(self.agent_locations[i])
            newAgent.no = i
            if weights is not None:
                newAgent.weight = weights[i]
            else:
                newAgent.weight = 1.0
            newAgent.type = type
            newAgent.init_demes(noAgents)
            self._agents.append(newAgent)


    def spawn_new_agents(self, noNew):
        start_i = self.noAgents
        for i in range(0, noNew):
            newAgent = eaAgent()
            newAgent.set_start(self.agent_locations[start_i + i])
            # newAgent.weight = (i+1) *0.5
            newAgent.no = start_i + i
            self._agents.append(newAgent)

    def spawn_defined_agents(self, locations):
        start_i = self.noAgents
        for i in range(0, len(locations)):
            newAgent = eaAgent()
            # print i,
            # print self.agent_locations[i]
            newAgent.set_start(locations[i])
            # newAgent.weight = (i+1) *0.5
            newAgent.no = start_i + i
            self._agents.append(newAgent)
