#!/usr/bin/env python3
from matplotlib import pyplot as plt
from matplotlib import gridspec
import string
import numpy as np
import logging
import subprocess
import time





def plot_global_progress(Results, state, foldername):
    colours = ['lightgreen', 'lightblue', 'coral', 'orange', 'mediumpurple', 'turquoise', 'olive', 'saddlebrown','plum', 'lightgrey']
    agent_names = string.ascii_uppercase
    fig = plt.subplots(figsize=(14, 14))
    # ax = [ax]
    ax0 = plt.subplot2grid((10, 10), (0, 0), rowspan = 1, colspan=10)
    ax1 = plt.subplot2grid((10, 10), (1, 0), rowspan = 1, colspan=10)
    ax2 = plt.subplot2grid((10, 10), (2, 0), rowspan = 2, colspan=10)
    ax3 = plt.subplot2grid((10, 10), (4, 0), rowspan = 5, colspan=5)
    ax4 = plt.subplot2grid((10, 10), (4, 5), rowspan = 5, colspan=5)


    ax3.set_xlim(state.xlimits)
    ax3.set_ylim(state.ylimits)
    pooled_scores = [res['pooled_score'] for res in Results]
    iterations = [res['Iteration'] for res in Results]
    noIterations = iterations[-1]
    simTimes = [res['simTime'] for res in Results]
    completedTasks = [res['CompletedTasks'] for res in Results]
    allCompletedTasks = [res['AllCompletedTasks'] for res in Results]
    noCompleted =  [len(res['AllCompletedTasks']) for res in Results]
    noTasks = [res['NoTasks'] for res in Results]
    distanceTravelled = [res['DistanceTravelled'] for res in Results]
    agentLocations = [res['AgentLocations'] for res in Results]
    exchanges = [res['pooled_exchange'] for res in Results]
    cumulativeDistance = np.sum(distanceTravelled, axis=1)
    exchangeMat = np.zeros((state.noAgents,state.noAgents), dtype=int)
    for exchange in exchanges:
        for item in exchange:

            exchangeMat[item[0]][item[1]] += 1
    agentDistances = []
    agentPaths = []
    for a in state.agentRange:
        dists = [dist[a] for dist in distanceTravelled]
        agentPaths.append([ [loc[a][0] for loc in agentLocations], [loc[a][1] for loc in agentLocations]])
        agentDistances.append(dists)

    ax0.plot(simTimes, noTasks, 'r-')
    ax0.plot(simTimes, noCompleted, 'g-')
    ax1.plot(simTimes, pooled_scores, 'g-')
    ax1.plot(simTimes, cumulativeDistance, 'r-')

    for a, agent in enumerate(state.agents):
        text_str = agent_names[agent.no]
        ax3.annotate(text_str, (agent.start_location[0], agent.start_location[1]), bbox={"boxstyle": "circle", "color": colours[a % len(colours)], "alpha": 0.25}, size=8)
        ax3.annotate(text_str, (agent.current_location[0], agent.current_location[1]), bbox={"boxstyle": "circle", "color": colours[a % len(colours)], "alpha": 1.0}, size = 8)

    for t, task in enumerate(state.tasks):
        # plt.plot(task.location[0], task.location[1], 'ko', markersize=10)
        text_str = '%i' %task.no
        # plt.text(task.location[0], task.location[1],text_str )
        alpha = 1.
        ax3.annotate(text_str, (task.location[0], task.location[1]),bbox={"boxstyle" : "round", "color": colours[task.completedBy % len(colours)], "alpha" : alpha},size=6)


    for a in state.agentRange:
        ax2.plot(simTimes, [b / a for b, a in zip(agentDistances[a], simTimes)],  '-', color=colours[a % len(colours)])
        ax3.plot(agentPaths[a][0], agentPaths[a][1],  '-', color=colours[a % len(colours)])

    ax2.set_ylim(bottom=0)
    ax4.imshow( exchangeMat/noIterations, cmap='summer', vmin=-.5, vmax=1.0, interpolation='nearest')
    for i in range(len(exchangeMat)):
        for j in range(len(exchangeMat)):
            text = ax4.text(j, i, round(exchangeMat[i, j]/noIterations, 3), ha="center", va="center", color="k")

    outputname = 'EATSP_summary'
    plt.savefig(foldername + '/' + outputname + '.png')
    logging.debug('plot saved to -> ' + foldername + '/' + outputname + '.png')
    plt.close()