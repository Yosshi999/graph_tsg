from main import NN, unit
import chainer
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
load_npz('trained_S', modelS)

modelG = NN(offset)
load_npz('trained_G', modelG)

models = [modelT, modelS, modelG]
time = np.linspace(0, 1, 30).astype(np.float32).reshape(-1, 1)


fig, ax = plt.subplots()
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
#plt.xlim(-2, 2)
#plt.ylim(2, -2)
artists_three = []

colors = 'rgby'

for model_i, model in enumerate(models):
    artists = [] # animation frames
    points = []
    for i in range(4):
        W = model.u[i].link.data
        W[np.where(abs(W) < 0.1)] = 0
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
        # for each units
        for pi, p in enumerate(points):
            base_x = p[0] + model_i * 2
            base_y = p[1]
            if abs(model.u[pi].omega.data[0, 0]) > 0.2:
                im.extend(ax.plot(base_x, base_y, '%c.'%colors[pi]))
            for j in range(model.u[pi].dim):
                if abs(model.u[pi].omega.data[j, 0]) > 0.2:
                    im.extend(ax.plot(base_x+p[2][:,j], base_y+p[3][:,j], '%c-'%colors[pi], linewidth=0.5))
                base_x += p[2][i, j]
                base_y += p[3][i, j]
                im.extend(ax.plot(base_x, base_y, '%c.'%colors[pi]))
            b_x.append(base_x)
            b_y.append(base_y)
        im.extend(ax.plot(
            [b_x[0], b_x[2]], [b_y[0], b_y[2]], 'k-',
            [b_x[1], b_x[3]], [b_y[1], b_y[3]], 'k-',
            predict_x.data[i], predict_y.data[i], "ko", linewidth=1, alpha=0.4))
        im.extend(ax.plot(predict_x.data[:], predict_y.data[:], "k-", alpha=0.5 ,linewidth=2))
        artists.append(im)

    artists_three.append(artists)
    print("ok")

artists = []
for i in range(len(artists_three[0])):
    ims = []
    for group in range(3):
        ims.extend(artists_three[group][i])
    artists.append(ims)

anim = ArtistAnimation(fig, artists, interval=200)

plt.show()

#plt.scatter(teacher[:, 0], teacher[:, 1], c="red")

#plt.show()