import networkx as nx
from matplotlib import pyplot as plt, animation
import numpy as np
from matplotlib.animation import FuncAnimation, writers, PillowWriter
#import matplotlib.animation as animation

from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython import display


alpha=1
beta=4
#Q=4
Q=240
pheramone_updates=0.33
pheramone_start=500
#N=10
N=50
#сделал +1 так как массивы должны быть больше, если стартуем с города 0?
cities = [[] for i in range(N)]
How_many_time_was_here = [[] for i in range(N)]
pheromone = [[] for i in range(N)]
attractiveness = [[] for i in range(N)]
for i in range(N):
    for j in range(i+1):
        cities[i].append(int(np.random.rand()*100+1))
        attractiveness[i].append(0)
        How_many_time_was_here[i].append(0)
        pheromone[i].append(pheramone_start)
        attractiveness[i][j]=1/cities[i][j]
def edge_selection(L,N,pheromones=0):

    chance = [[] for i in range(N)]

    #chance table
    for i in range(N):
        for j in range(i + 1):
            chance[i].append(1)

    #delete move chances L-1
    for j in range(len(L)-1):
        for i in range(L[j],N):
            chance[i][L[j]]=0
        for k in range(L[j]):
            chance[L[j]-1][k]=0

    #chance x->y and chance table
    all_chance = 0
    start_point=L[-1] #4

    for i in range(start_point):
        all_chance+=pheromone[start_point-1][i]*attractiveness[start_point-1][i]*chance[start_point-1][i]
    for j in range(start_point,N):
        all_chance+=pheromone[j][start_point]*attractiveness[j][start_point]*chance[j][start_point]

    # print(chance)
    chance_table=[]

    for i in range(start_point):
        chance[start_point-1][i] = pheromone[start_point-1][i]*attractiveness[start_point-1][i]*chance[start_point-1][i]/all_chance
        chance_table.append(chance[start_point-1][i])
    for j in range(start_point, N):
        chance[j][start_point] = pheromone[j][start_point]*attractiveness[j][start_point]*chance[j][start_point]/all_chance
        chance_table.append(chance[j][start_point])
    return(chance_table)
def move(N):
    F=[]
    for i in range(N+1):
        L=[i]
        for j in range(N):
            #drop - массив вероятностей
            drop=edge_selection(L, N)
            k = 0
            summa = 0
            #бросок кубика
            dice = np.random.rand()
            while summa<dice:
                summa+=drop[k]
                k+=1
            if k<=L[-1]:
                L.append(k-1)
            else:
                L.append(k)
        F.append(L)
    return(F)

def pherameone_update(N,E,cities):

    #print("pheromone", pheromone)

    for i in range(N):
        for j in range(len(pheromone[i])):
            pheromone[i][j]*=(1-pheramone_updates)

    #print("pheromone2", pheromone)
    #print("E", E)

    dim=[0 for i in range(N+1)]
    for i in range(N+1):
        for j in range(N+1):
            if (E[i][j-1]<E[i][j]):
                dim[i]+=cities[E[i][j]-1][E[i][j-1]]
            else:
                dim[i] += cities[E[i][j-1]-1][E[i][j]]

    #print("DIM", dim)

    for i in range(N + 1):
        for j in range(N + 1):
            if (E[i][j - 1] < E[i][j]):
                pheromone[E[i][j]-1][E[i][j-1]]+=Q/dim[i]
            else:
                pheromone[E[i][j-1]-1][E[i][j]]+=Q/dim[i]

    #print("pheromone3", pheromone)

    return(dim,E)

labels = {}
for i in range(N+1):
    labels[i]=i

G=nx.Graph()
#G=nx.grid_2d_graph(4,4)

#fig = plt.figure()
fig = plt.figure("Degree of a random graph", figsize=(8, 8))

#fig, _ = plt.subplots(figsize=(10, 5))
#G = plt.subplot(221)
#ax1 = plt.subplot(222)

#fig, all_axes = plt.subplots(2, 2)
#ax = all_axes.flat



nodes=[]
for i in range(N+1):
    nodes.append(i)
#print(nodes)
edges= []
for i in range(N):
    for j in range(i+1):
        edges.append([i+1,j])

G.add_nodes_from(nodes)
G.add_edges_from(edges)

#pos = nx.random_layout(G)
pos = nx.circular_layout(G)
options = {"edgecolors": "tab:gray", "node_size": 300, "alpha": 0.9}

fig.set_facecolor("black")

#DEWFWE
A = nx.gnp_random_graph(100, 0.02, seed=10374196)

degree_sequence = sorted((d for n, d in A.degree()), reverse=True)

axgrid = fig.add_gridspec(5, 4)

ind = 0
x = []
y = []
glob_min=10000
edges3 = []
def animate(frame):
    print(frame)
    check_upd=False
    global edges3
    global ind
    global glob_min

    fig.clear()

    fig.set_facecolor("black")
    plt.gca().set_facecolor('black')
    ax0 = fig.add_subplot(axgrid[0:3, :])
    T = move(N)
    dims,dist=pherameone_update(N, T, cities)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color="tab:blue",ax=ax0, **options)
    nx.draw_networkx_labels(G, pos, labels, font_size=5, font_color="whitesmoke",ax=ax0)
    for i in range(len(pheromone)):
        for j in range(len(pheromone[i])):
            edges2=[]
            edges2.append([i+1,j])
            nx.draw_networkx_edges(G, pos, edgelist=edges2, width=4, alpha=pheromone[i][j]/500, edge_color="tab:blue", ax=ax0)


    ax0.set_title("Ant colony optimization algorithms, visualization of pheromones", color="w")
    ax0.set_axis_off()


    minimum=min(dims)
    if minimum<glob_min:
        glob_min=minimum
        check_upd=True

    #print("min", minimum)

    ind+=1
    x.append(ind)
    y.append(minimum)

    #print("x",x)
    #print("y",y)

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(x,y, "b-", marker="o", label=f"current {y[-1]}; global {glob_min}")
    ax1.set_title("shortest distance", color='w')
    ax1.set_ylabel("y, distance",color='w')
    ax1.set_xlabel("x, iteration",color='w')
    ax1.spines['bottom'].set_color('yellow')
    ax1.spines['left'].set_color('yellow')
    ax1.tick_params(axis='x', colors='yellow')  # setting up X-axis tick color to red
    ax1.tick_params(axis='y', colors='yellow')  # setting up Y-axis tick color to black
    ax1.grid()
    ax1.legend(loc="upper right")


    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.set_title("The right way", color="w")
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color="tab:blue",ax=ax2, **options)
    nx.draw_networkx_labels(G, pos, labels, font_size=5, font_color="whitesmoke",ax=ax2)
    for i in range(len(pheromone)):
        for j in range(len(pheromone[i])):
            edges2=[]
            edges2.append([i+1,j])
            nx.draw_networkx_edges(G, pos, edgelist=edges2, width=4, alpha=pheromone[i][j]/500, edge_color="tab:blue", ax=ax2)
    way=dims.index(minimum)

    #print("dist",dist)

    if check_upd==True:
        edges3 = []
        for i in range(N+1):
            edges3.append([dist[way][i-1],dist[way][i]])

    nx.draw_networkx_edges(G, pos, edgelist=edges3, width=4, alpha=1, edge_color="tab:red",ax=ax2)

ani = animation.FuncAnimation(fig, animate, frames=60, interval=1000, repeat=True,save_count=4000)


plt.tight_layout()
plt.axis("off")

"""
writer = PillowWriter(fps=25)
ani.save("3_nodes.gif", writer=writer)
"""

"""
writervideo = animation.FFMpegWriter(fps=60)
ani.save('increasingStraightLine1.mp4', writer=writervideo)
"""

#f = "animation125.gif"
#writergif = animation.PillowWriter(fps=30)
#ani.save(f, writer=writergif)

#ani.save('animation.mp4', progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'))

#ani.save('growingCoil.mp4', writer = 'ffmpeg', fps = 30)
progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')

ani.save("animtest6.gif",writer = 'ffmpeg',fps=None,progress_callback=None)

plt.show()