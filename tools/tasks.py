from tools.assignment_tools import distance, generate_samples
import random
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from tools.hexes import Point
import logging

class Task():
    def __init__(self, location = [0.0,0.0]):
        self.location = location
        self.complete = False
        self.completedBy = -1
        self.completedAt = 0.0
        self.spawnedAt = 0.0
        self.no = 0
        self.type = 0

    def set_location(self, location):
        self.location = location

    def completed(self, completedBy, t):
        self.complete = True
        self.completedBy = completedBy
        self.completedAt = t


class Tasker():
    def __init__(self, type='RandomList'):
        self.type = type


class randomTasker(Tasker):
    def __init__(self):
        super(Tasker, self).__init__()

        samp_dist = 200
        samp_radius = 5
        self.locations = generate_samples(samp_dist, samp_radius, 10)
        # s = np.where(X > 0)
        # self.locations = [[float(s[0][i]), float(s[1][i])] for i in range(0, len(s[0]))]
        random.shuffle(self.locations)

    def spawn_task(self, i):
        newTask = Task()
        newTask.set_location(self.locations[i])
        newTask.no = i

        return newTask


class blobTasker(Tasker):
    def __init__(self, noBlobs=3):
        super(Tasker, self).__init__()
        X = generate_samples(200, 50, 10)
        self.centers = X[:noBlobs] #[[10., 10.], [150., 25.], [100., 175.]]
        self.samples = 200# [150, 75, 75]
        self.xlim = [0, 200]
        self.ylim = [0, 200]
        self.std = 30 # std dev of points

        self.make_clusters()

    def make_clusters(self):
        points, cluster = make_blobs(n_samples=self.samples, centers=self.centers, cluster_std=self.std, shuffle=True)
        # self.locations = points.tolist()
        self.locations = [[point[0], point[1]] for point in points  if point[0] >= self.xlim[0] and point[1] >= self.ylim[0] and point[0] < self.xlim[1] and point[1] < self.ylim[1]]
        random.shuffle(self.locations)

    def spawn_task(self, i):
        newTask = Task()
        newTask.set_location(self.locations[i])
        newTask.no = i

        return newTask

class hexTasker(Tasker):
    def __init__(self, hexGrid):
        super(Tasker, self).__init__()
        self.hex_grid = hexGrid
        self.xlim = [0, 200]
        self.ylim = [0, 200]


    def spawn_task(self, i):
        selMethod = 'tournament'
        newTask = Task()
        taskOK = False
        if selMethod == 'tournament':
            count = 0
            best_score = 1e9
            best_loc = None
            # while not taskOK and (count < 10):
            N = 20 # no samples
            locs = [[random.uniform(self.xlim[0], self.xlim[1]), random.uniform(self.ylim[0], self.ylim[1])] for i in range(0, N)]
            scores = []
            for loc in locs:
                hex = self.hex_grid.layout.pixel_to_hex(Point(loc[0], loc[1]))
                score = self.hex_grid.hexinfo[(hex.q, hex.r)]['score'] #higher score means visited more, so want to favour lower scores.
                scores.append(score)
                #
                # P0 = (1./(score*score + 1))
                # p = random.random()
                # count += 1
                # if p <= P0:
                #     print("score: %2.2f, P0: %2.2f, p: %2.2f" %(score, P0, p))
                #     taskOK = True
            best_loc = np.argmin(scores)
            # logging.info("Taking loc %i with score %2.2f" %(best_loc, scores[best_loc]))
            loc = locs[best_loc]
        else:
            loc = [random.uniform(self.xlim[0], self.xlim[1]), random.uniform(self.ylim[0], self.ylim[1])]

        newTask.set_location(loc)
        newTask.no = i

        return newTask