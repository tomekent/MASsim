from tools.assignment_tools import distance

class Agent():
    def __init__(self, type = 0):
        # initialise agent
        self.start_location = [0.0, 0.0]
        self.current_location = [0.0, 0.0]
        self.weight = 1.0 # cost weight
        self.no = 0
        self.type = type
        self.authority = 0
        self.maxCommsRadius = 0.0
        self.commsRadius = 0.0
        self.v = 1.0
        self.distanceTravelled = 0.0
        self.loiterScheme = 'None'
        self.currentRoute = []
        self.completedTasks = []
        self.locationHistory = []
        self.taskCapabilities = set([0])

    def setCapabilities(self, taskTypes=[]):
        for taskType in taskTypes:
            self.taskCapabilities.add(taskType)

    def set_start(self, location):
        self.start_location = [location[0], location[1]]
        self.current_location = location

    def distance_to_target(self, target):
        return distance(self.current_location, target)

    def move_toward(self, target, v, dt):
        if target is None:
            return
        delta = v*dt
        dist = self.distance_to_target(target)

        if dist < 0.1:
            ndx = 0.0
            ndy = 0.0
        else:
            ndx = (target[0] - self.current_location[0]) / dist
            ndy = (target[1] - self.current_location[1]) / dist

        if delta > dist:
            delta = dist

        self.current_location[0] += delta * ndx
        self.current_location[1] += delta * ndy
        self.distanceTravelled += delta

    def move_direction(self, direction, v, dt):
        delta = v * dt

        self.current_location[0] += direction[0]*delta
        self.current_location[1] += direction[1]*delta



    def degradeComms(self, factor=0.0):
        self.commsRadius = max(min(self.commsRadius - self.maxCommsRadius*factor, self.maxCommsRadius),0.0)
        print('Degrading comms by %2.2f to %2.2f' %(factor, self.commsRadius))

    def improveComms(self, factor=0.0):
        self.commsRadius = max(min(self.commsRadius + self.maxCommsRadius*factor, self.maxCommsRadius),0.0)
        print('Improving comms by %2.2f to %2.2f' %(factor, self.commsRadius))

class eaAgent(Agent):
    def __init__(self):
        super().__init__()
        self.demes = []

    def init_demes(self, noAgents):
        self.demes = [[] for agent in range(noAgents)]

    # def remove_task_from_demes(self, taskNo):
    #     for deme in self.demes:
    #         for ind in deme:
    #             for route in ind.Routes:
    #                 if taskNo in route:
    #                     route.remove(taskNo)
    #                     ind.noTasks -= 1
    #                     break
    #             ind.resetKnown(ind.Routes)

    def pool_demes(self):
        pool = []
        for deme in self.demes:
            pool.extend(deme.members)
        return pool
