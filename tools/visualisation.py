#!/usr/bin/env python3
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import colors as cls
from matplotlib import cm
import string
import numpy as np
import logging
import subprocess
import time

def plot_setup(state, ngen, commsRadius=False, considerationRadius=0):
    # plotting stuff
    taskcolours = ['lightgrey', 'lightgreen', 'plum', 'lightblue']
    colours = ['lightgreen', 'lightblue', 'coral', 'orange' ,'mediumpurple', 'turquoise','olive','saddlebrown','plum','lightgrey']
    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax = [ax]
    # ax[0] = plt.subplot2grid((5, 3), (0, 0), colspan=3, rowspan=3)
    # ax[1] = plt.subplot2grid((5, 3), (3, 0), colspan=3, rowspan=2)
    # gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    lines = [[],[],[]]
    lines[0] = [ax[0].plot([], [],'-' ,color=colours[a % len(colours)])[0] for a in state.agentRange]
    # lines[1] = [ax[0].plot([], [],':' ,color=colours[a % len(colours)])[0] for a in state.agentRange]
    # lines[2] = [ax[1].plot([], [],'x',color='blue', markersize=1)[0], ax[1].plot([], [],'*',color='red', markersize=1)[0]]
    for t, task in enumerate(state.tasks):
        # plt.plot(task.location[0], task.location[1], 'ko', markersize=10)
        text_str = '%i' %task.no
        # plt.text(task.location[0], task.location[1],text_str )
        if task.complete:
            tdiff = state.simTime - task.completedAt
            if tdiff <= 10: # fade out completed tasks
                alpha = 1./(1. + tdiff)
                # ax[0].annotate(text_str, (task.location[0], task.location[1]),bbox={"boxstyle" : "round", "color": colours[task.completedBy % len(colours)], "alpha" : alpha},size=6)
        else:
            alpha = 1.
            ax[0].annotate(text_str, (task.location[0], task.location[1]), bbox={"boxstyle": "round", "color": taskcolours[task.type % len(colours)],"alpha": alpha}, size=6)
            # ax[0].annotate(text_str, (task.location[0], task.location[1]),bbox={"boxstyle" : "round", "color": "lightgrey"},size=6)


    agent_names = string.ascii_uppercase
    for a in state.agentRange:
        agent = state.agents[a]
        if state.noAgents > len(agent_names):
            text_str = agent_names[agent.no % len(agent_names)] + str(int(agent.no/len(agent_names)))
        else:
            text_str = agent_names[agent.no]
        # if commsRadius:
        #     ax[0].add_patch(plt.Circle((agent.current_location[0], agent.current_location[1]), agent.commsRadius, color='r', alpha=0.025))
        # if considerationRadius >0:
        #     ax[0].add_patch(plt.Circle((agent.current_location[0], agent.current_location[1]), agent.commsRadius + considerationRadius, color='g', alpha=0.0125))
        ax[0].annotate(text_str,(agent.current_location[0], agent.current_location[1]),bbox={"boxstyle" : "circle", "color": colours[a % len(colours)]},size=8)

    ax[0].set_xlim([0,100])
    ax[0].set_ylim([0,100])
    # ax[1].set_xlim([0, ngen])
    # ax[1].set_ylim([0, 5000])

    return fig, ax, lines

def plot_update(Routes, gen, gens, mins, avgs, popscores, offscores, state, ax, lines, newTitle=None):
    # if gen == 1:
    #     maxy = (int(avgs[0]/50)+5)*50
    #     ax[1].set_ylim([0,maxy])
    routes = []
    for a in state.agentRange:
        fullroute = [a - state.noAgents] + Routes[a] + [a - state.noAgents]
        route = []
        for i in range(0, len(fullroute)-1):
            route.append([fullroute[i], fullroute[i+1]])
        routes.append(route)

    for a in state.agentRange:
        xs = []
        ys = []
        for arci in range(0, len(routes[a])):
            # print arci
            arc = routes[a][arci]
            # print real_routes[a]
            if arc[0]<0:
                loc_i = state.agents[arc[0]+state.noAgents].current_location
            else:
                loc_i = state.tasks[arc[0]].location
            if arc[1]<0:
                loc_j = state.agents[arc[1]+state.noAgents].current_location
                lstyle = ':'
                lwidth = 1.0
            else:
                loc_j = state.tasks[arc[1]].location
                lstyle = '-'
                lwidth = 3.0
            xs.append(loc_i[0])
            ys.append(loc_i[1])
        # xs.append(loc_j[0])
        # ys.append(loc_j[1])
        lines[0][a].set_data(xs, ys)
    if newTitle is None:
        title = 'Generation %i' %(gen)
        # res_title = 'Obj: %2.2f' %(obj)
        ax[0].set_title(title)# +'\n' +res_title )
    else:
        ax[0].set_title(newTitle )

    #
    # lines[1][0].set_data(gens, mins)
    # lines[1][1].set_data(gens, avgs)

    # pop_xs = lines[2][0].get_xdata()
    # pop_ys = lines[2][0].get_ydata()
    # pop_xs = np.append(pop_xs, [gen] * len(popscores))
    # pop_ys = np.append(pop_ys, popscores)
    # off_xs = lines[2][1].get_xdata()
    # off_ys = lines[2][1].get_ydata()
    # off_xs = np.append(off_xs, [gen-0.25] * len(offscores))
    # off_ys = np.append(off_ys, offscores)
    # lines[2][0].set_data(pop_xs, pop_ys)
    # lines[2][1].set_data(off_xs, off_ys)
    # lines[1][2].set_data(gens, maxs)
    #
    plt.pause(0.005)
    # fig.canvas.draw()
    return lines,

def plot_close():
    plt.close()

def plot_save(foldername, state, suffix=''):
    outputname = 'EATSP_result'
    outputname += suffix
    plt.savefig(foldername + '/' + outputname +'.png')
    logging.debug('plot saved to -> ' +foldername + '/' + outputname +'.png')


def plot_result(state, Routes, pooled_exchange = [], logbook=None, foldername='', save_plot=False, suffix='', commsRadius=False, considerationRadius=0.0):
    if logbook is None:
        gens, avgs, mins = [], [], []
        ngen = 0
    else:
        gens, avgs, mins = logbook.select("gen", "avg", "min")
        ngen = gens[-1]

    routes = []

    for a in state.agentRange:
        fullroute = [a - state.noAgents] + Routes[a] + [a - state.noAgents]
        route = []
        for i in range(0, len(fullroute)-1):
            route.append([fullroute[i], fullroute[i+1]])
        routes.append(route)

    #### PLOT
    # foldername = 'EATSP'
    colours = ['lightgreen', 'lightblue', 'coral', 'orange' ,'mediumpurple', 'turquoise','olive','saddlebrown','plum','lightgrey']
    # fig, ax = plt.subplots(figsize=(12, 8))

    # xdata, ydata = [], []
    # lines = [plt.plot([], [],'-' ,color=colours[a % len(colours)])[0] for a in range(noAgents)]

    fig, ax, lines = plot_setup(state, ngen, commsRadius=commsRadius, considerationRadius=considerationRadius)

    # color_map = plt.cm.Spectral_r
    # allLocs = []
    # for agent in state.agents:
    #     allLocs.extend(agent.locationHistory)
    #     # for loc in agent.locationHistory:
    #     #     np.append(allLocs, loc)
    # tmin = state.simTime - 60.0
    # if len(allLocs) > 0:
    #     allLocs = np.array(allLocs)
    #     extent = state.xlimits[:] + state.ylimits[:]
    #     indt = np.argwhere(allLocs[:,0] > tmin)
    #
    #     if len(indt) > 0:
    #         hexplt = plt.hexbin(allLocs[indt,1], allLocs[indt,2], cmap='Greens',gridsize=10,clim=(0, 20), extent=extent,mincnt=1)
            # cb = plt.colorbar(hexplt, spacing='uniform', extend='max')
            # counts = hexplt.get_array()
            # ncnts = np.count_nonzero(np.power(10, counts))
            # verts = hexplt.get_offsets()
            # for offc in range(verts.shape[0]):
            #     binx, biny = verts[offc][0], verts[offc][1]
            #     if counts[offc]:
            #         plt.plot(binx, biny, 'k.', zorder=100)

            # fig.set_size_inches(8, 7)
        # print routes
    # normalize item number values to colormap
    norm = cls.Normalize(vmin=0, vmax=10)
    hexes = []
    total_score = 0.0
    total_alive = 0
    for hexkey, hexi in state.hex_grid.hexes.items():
        corners = state.hex_grid.layout.polygon_corners(hexi)
        points = [[corner.x, corner.y] for corner in corners]
        score = state.hex_grid.hexinfo[hexkey]['score']
        if score >0:
            total_alive += 1
        total_score += score
        if state.hex_grid.hexinfo[hexkey]['last_visited'] > 0:
            tdiff = state.simTime - state.hex_grid.hexinfo[hexkey]['last_visited']

            # if tdiff <= 100:  # fade out completed tasks
            #     alpha = 1. / (1. + tdiff)
            #     hexpatch = plt.Polygon(points, fill=True, color='lightgreen', alpha=alpha)
            # else:
            cl = cm.RdYlGn(norm(score),bytes=True)
            col = [x/255 for x in cl[:3]]
            hexpatch = plt.Polygon(points, fill=True, color=col, alpha=0.5)


        else:
            hexpatch = plt.Polygon(points, fill=True, color='lightgrey', alpha=0.25)
        center = state.hex_grid.layout.hex_to_pixel(hexi)
        ax[0].add_patch(hexpatch)
        ax[0].annotate('%2.2f' % (score), (center.x, center.y), color='w', weight='bold',
                       fontsize=6, ha='center', va='center')

    for a in state.agentRange:
        xs = []
        ys = []
        for arci in range(0, len(routes[a])):
            # print arci
            arc = routes[a][arci]
            # print real_routes[a]
            if arc[0]<0:
                loc_i = state.agents[arc[0]+state.noAgents].current_location
            else:
                loc_i = state.tasks[arc[0]].location
            if arc[1]<0:
                loc_j = state.agents[arc[1]+state.noAgents].current_location
                lstyle = ':'
                lwidth = 1.0
            else:
                loc_j = state.tasks[arc[1]].location
                lstyle = '-'
                lwidth = 3.0
            xs.append(loc_i[0])
            ys.append(loc_i[1])
            lines[0][a].set_data(xs, ys)
            # lines, = plt.plot([loc_i[0], loc_j[0]], [loc_i[1], loc_j[1]], lstyle ,color=colours[a % len(colours)], linewidth=lwidth)

    for exchange in pooled_exchange:
        loc1 = state.agents[exchange[0]].current_location
        loc2 = state.agents[exchange[1]].current_location
        ax[0].plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], ':', color='green')
    title = 'No Agents: %i, no Tasks: %i, hexScore: %2.2f, alive: %i' %(state.noAgents, state.noTasks, total_score, total_alive)
    # res_title = 'Obj: %2.2f' %(obj)
    ax[0].set_title(title) # +'\n' +res_title )
    max_locs = [0.0, 0.0]

    ax[0].set_xlim(state.xlimits)
    ax[0].set_ylim(state.ylimits)


    # plot the line graph

    # if len(avgs)>0:
    #     maxy = (int(avgs[0]/50)+1)*50
    #     ax[1].set_ylim([0,maxy])
    # lines[1][0].set_data(gens, mins)
    # lines[1][1].set_data(gens, avgs)
    # plt.show()
    if save_plot:
        plot_save(foldername, state, suffix=suffix)
        plot_close()
    else:
        # plot_save(foldername, noAgents, noTasks, suffix=suffix)
        plt.show(block=False)
        plt.pause(0.015)

    return ax, lines


def convert_to_video(foldername):
    logging.info('Converting output images to video...')
    subprocess.call('pwd')
    subprocess.call("./EATSP/create_video.sh %s" %foldername,  shell=True)
    logging.info('Conversion Complete.')


