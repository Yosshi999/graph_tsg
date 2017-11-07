from main import NN, unit
import chainer
import chainer.functions as F
from chainer.serializers import load_npz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import sys

offset = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1]
]
modelT = NN(offset)
load_npz('trained_T', modelT)

modelS = NN(offset)
load_npz('trained_20171107S', modelS)

modelG = NN(offset)
#load_npz('trained_G', modelG)
load_npz('trained_20171101_1G', modelG)

models = [modelT, modelS, modelG][:]
time = np.linspace(0, 1, 40).astype(np.float32).reshape(-1, 1)


fig, ax = plt.subplots()
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
#plt.xlim(-2, 2)
#plt.ylim(2, -2)
ax.tick_params(labelbottom="off",bottom="off") # x軸の削除
ax.tick_params(labelleft="off",left="off") # y軸の削除
ax.set_xticklabels([]) 
plt.box("off") #枠線の削除

artists_three = []

color_list = ['rgby', 'rgby', 'rgby']

for model_i, model in enumerate(models):
    colors = color_list[model_i]
    artists = [] # animation frames
    points = []
    for i in range(4):
        W = model.u[i].link.data
        W[np.where(abs(W) < 0.2)] = 0
        model.u[i].link = chainer.Parameter(initializer=W, shape=(1, model.u[i].dim+2))
        
        points.append( model.u[i].calc(time) )
        """plt.plot(center_x.flatten(), center_y.flatten())
        for j in range(model.u[i].dim):
            plt.plot(satellite_x[:,j], satellite_y[:,j])"""
    predict_x, predict_y = model.calc(time)
    predict_x += model_i * 2

    for i in range(len(time)):
        im = []
        b_x = []
        b_y = []
        """if not i == 48:
            continue"""
        # for each units
        for pi, p in enumerate(points):
            base_x = p[0] + model_i * 2
            base_y = p[1]
            if abs(model.u[pi].omega.data[0, 0]) > 0.2:
                im.extend(ax.plot(base_x, base_y, '%c.'%colors[pi]))
                #im.extend(ax.plot(base_x, base_y, color='0.3', marker='.'))
            for j in range(model.u[pi].dim):
                omega = abs(F.floor(model.u[pi].omega).data[j, 0])
                if omega > 0:
                    end = len(time)//int(omega) + 1
                    #end = len(time)
                    im.extend(ax.plot(base_x+p[2][:end,j], base_y+p[3][:end,j], '%c-'%colors[pi], linewidth=0.5))
                    #im.extend(ax.plot(base_x+p[2][:end,j], base_y+p[3][:end,j], color='0.3', linewidth=0.5))
                base_x += p[2][i, j]
                base_y += p[3][i, j]
                im.extend(ax.plot(base_x, base_y, '%c.'%colors[pi]))
                #im.extend(ax.plot(base_x, base_y, color='0.3', marker='.'))
            b_x.append(base_x)
            b_y.append(base_y)
        
        im.extend(ax.plot(
            [b_x[0], b_x[2]], [b_y[0], b_y[2]],
            [b_x[1], b_x[3]], [b_y[1], b_y[3]], '-', linewidth=1, color='0.7'))
        im.extend(ax.plot(predict_x.data[i], predict_y.data[i], color='0.4', marker='o'))
        im.extend(ax.plot(predict_x.data[:], predict_y.data[:], '-', color='0', linewidth=2))
        artists.append(im)
        """
        im.extend(ax.plot(
            [b_x[0], b_x[2]], [b_y[0], b_y[2]], '%c-'%colors[0],
            [b_x[1], b_x[3]], [b_y[1], b_y[3]], '%c-'%colors[0], linewidth=1, alpha=0.3))
        im.extend(ax.plot(predict_x.data[i], predict_y.data[i], '%co'%colors[0], alpha=0.6))
        im.extend(ax.plot(predict_x.data[:], predict_y.data[:], '%c-'%colors[0], linewidth=2))
        artists.append(im)
        """
    artists_three.append(artists)
    print("ok")

artists = []
for i in range(len(artists_three[0])):
    ims = []
    for group in range(len(models)):
        ims.extend(artists_three[group][i])
    artists.append(ims)

anim = ArtistAnimation(fig, artists, interval=200)
#anim.save('out2.mp4', writer='ffmpeg')
plt.show()
