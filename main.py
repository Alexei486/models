import matplotlib.axes
import numpy as np
import matplotlib as mlib
from matplotlib import colors
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation, writers, PillowWriter
import matplotlib.animation as animation


# cube function

def cube1(T=0):
    T = 0
    return (int(np.random.rand() * 6) + 1)


def cube2(T=0):
    T = 0
    return (6)


def cube3(T=10):
    mu = 6
    sigma = T
    N = 1000
    number = np.random.normal(mu, sigma, N)
    x = []
    y = []
    for i in range(N):
        if (1 <= number[i] and number[i] <= 7):
            y.append(int(number[i]))
    for i in range(len(y)):
        x.append(i)
    return (y[0])


#gaus_spam
def cube3GA(T=10):
    mu = 6
    sigma = T
    N = 10000
    number = np.random.normal(mu, sigma, N)
    x = []
    y = []
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    for i in range(N):
        if (1 <= number[i] and number[i] <= 7):
            y.append(int(number[i]))
    for i in range(len(y)):
        x.append(i)
    for i in y:
        if i==1:
            a+=1
        if i==2:
            b+=1
        if i==3:
            c+=1
        if i==4:
            d+=1
        if i==5:
            e+=1
        if i==6:
            f+=1
        a /= (a + b + c + d + e + f)
        b /= (a + b + c + d + e + f)
        c /= (a + b + c + d + e + f)
        d /= (a + b + c + d + e + f)
        e /= (a + b + c + d + e + f)
        f /= (a + b + c + d + e + f)
    return (a,b,c,d,e,f)


# map
map = []
for i in range(113):
    map.append(i)
gold = [4, 34, 49, 63, 74, 91, 101]
red = [6, 30, 50, 60, 76, 100, 110]
# red=[-1]
red_ban = [-2, -2, -2, -3, -3, -2, -3]
green = [10, 25, 43, 57, 72, 81, 95, 106]
green_ban = [+3, +2, +2, +1, +2, +3, +2, +3]
black = [13, 33, 46, 55, 70, 85, 103]
# black=[-1]

ade = np.zeros((6, 19))

for i in range(114):
    m = int(i / 19)
    k = i % 19
    ade[m][k] = 5
    for j in gold:
        if i == j:
            ade[m][k] = (25)
    for j in green:
        if i == j:
            ade[m][k] = (15)
    for j in red:
        if i == j:
            ade[m][k] = (35)
    for j in black:
        if i == j:
            ade[m][k] = (45)

# print(ade)

''''''


# event_true_or_false
def event_check(coord):
    for i in gold:
        if coord == i:
            return 1
    for i in red:
        if coord == i:
            return 1
    for i in green:
        if coord == i:
            return 1
    for i in black:
        if coord == i:
            return 1
    else:
        return 0


# event
def event(coord, moves, cube, T=10):
    for i in gold:
        if (coord == i):
            coord += cube(T)
            return (coord, moves)
    for ind, val in enumerate(red):
        if (coord == val):
            coord += red_ban[ind]
            return (coord, moves)
    for ind, val in enumerate(green):
        if (coord == val):
            coord += green_ban[ind]
            return (coord, moves)
    for i in black:
        if (coord == i):
            moves += 2
            coord += cube(T)
            return (coord, moves)


# game_player
def game(cube, T=10):
    player1_coord = 0
    moves = 0
    moves_all = []
    while (player1_coord <= 113):
        player1_coord += cube(T)
        moves_all.append(player1_coord)
        moves += 1
        while (event_check(player1_coord) == 1):
            player1_coord, moves = event(player1_coord, moves, cube, T)
            event_check(player1_coord)
            moves_all.append(player1_coord)
    return (moves, moves_all)


# mean of games 1
def start1(T_start=10):
    mass = []
    for i in range(100):
        T = 10
        a, moves_alls = game(cube1, T)
        mass.append(a)
    summa = np.mean(mass)
    print(summa)
    return (summa, moves_alls, a)


# mean of games 2
def start2(T_start=10):
    mass2 = []
    for i in range(100):
        T = T_start
        a2, moves_alls2 = game(cube3, T)
        mass2.append(a2)
    summa2 = np.mean(mass2)
    return (summa2, moves_alls2, a2)


start2(T_start=0.001)

fig, _ = plt.subplots(figsize=(10, 5))

ax0 = plt.subplot(221)
ax1 = plt.subplot(223)
ax2: matplotlib.axes.Axes = plt.subplot(222)
ax3: matplotlib.axes.Axes = plt.subplot(224)

# fig, (ax0, ax1) = plt.subplots(nrows=2)

def plot_colored_grid(data, ax, colors=['white', 'green'], bounds=[0, 0.5, 1], grid=True, labels=False, frame=True):
    # create discrete colormap
    cmap = mlib.colors.ListedColormap(colors)
    norm = mlib.colors.BoundaryNorm(bounds, cmap.N)
    # show grid

    if grid:
        ax.grid(axis='both', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, data.shape[1], 1))  # correct grid sizes
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1))

    # disable labels
    if not labels:
        ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # plot data matrix
    ax.imshow(data, cmap=cmap, norm=norm)


data = ade
x = []
y = []
# fig = plt.figure()


# !!!! fig, (ax0, ax1) = plt.subplots(nrows=2)

cube_normal = start1(T_start=0.001)
'''
cube_gauss = start2(T_start=0.001)
'''
# gauss_many

cube_gausses=[]
gausses_moves=[]
spam_gauss=[]

for i in range(1,20):
    print(i)
    cube_gauss = start2(T_start=100*(0.7**i))
    gausses_moves.append(cube_gauss[0])
    cube_gausses.append(cube_gauss[1])
    spam_gauss.append(cube3GA(100*(0.7**i)))


gr0 = ax0.plot([], [], "o")[0]
gr1 = ax1.plot([], [], "o")[0]
gr2 = ax2.plot([], [], "ro-")[0]
gr3 = ax3.plot([], [], "ro-")[0]

ax2.set_ylim(0, 20)
ax2.set_xlim(0, 20)

ax3.set_ylim(0, 1)
ax3.set_xlim(0, 6)

ax2.set_title("разность ходов")
ax3.set_title("распределение вероятностей")

nemo=0
j2=0
x = []
y = []
x1=[]
y1=[]
x2=[]
y2=[]
x3=[]
y3=[]
spam1=[1,2,3,4,5,6]
size_gr=0
for i in range(len(cube_gausses)):
    size_gr+=len(cube_gausses[i])
print(size_gr)

def animation_func(i):
    print(i)
    if (i<size_gr):
        global j2
        global nemo
        j2 +=1

        if (j2 <= max(len(cube_normal[1]), len(cube_gauss[1]))):
            if (j2 < len(cube_normal[1]) - 1):
                ax0.set_title(f"Moves: {j2}; Temperature {100}")
                m = int(cube_normal[1][j2] / 19)
                x.append(cube_normal[1][j2] % 19)
                y.append(0 + m)
                gr0.set_data(x, y)

            if (j2 < len(cube_gausses[nemo]) - 1):
                T= 10*(0.7**(nemo+1))

                ax1.set_title(f"Moves: {j2}; Temperature {T}")
                m = int(cube_gausses[nemo][j2] / 19)
                x1.append(cube_gausses[nemo][j2] % 19)
                y1.append(0 + m)
                gr1.set_data(x1, y1)
        else:
            gr3.set_data(spam1, spam_gauss[nemo])

            different=abs(33-gausses_moves[nemo])
            x2.append(nemo)
            y2.append(different)
            gr2.set_data(x2,y2)
            j2 = 0
            nemo+=1
            x.clear()
            y.clear()
            x1.clear()
            y1.clear()
            gr0.set_data(x, y)
            gr1.set_data(x, y)
        #ax0.scatter(x, y)
        #ax1.scatter(x1, y1)


plot_colored_grid(data, ax0, colors=['white', 'green', 'yellow', 'red', 'black'], bounds=[0, 10, 20, 30, 40, 50])
plot_colored_grid(data, ax1, colors=['white', 'green', 'yellow', 'red', 'black'], bounds=[0, 10, 20, 30, 40, 50])

animation = FuncAnimation(fig, animation_func,
                          interval=100, frames=size_gr-1, repeat=False)

ax2.grid()
ax3.grid()
''''''
#animation.save("anim7.gif")
#video = animation.to_html5_video()
'''
f = r"animatione.gif"
writergif = animation.PillowWriter(fps=30)
animation.save(f, writer=writergif)
'''


writer = PillowWriter(fps=25)
animation.save("demo_sine6.gif", writer=writer)

plt.show()