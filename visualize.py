from main import NN, unit
import chainer
import chainer.functions as F
from chainer.serializers import load_npz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import sys
import matplotlib.transforms

offset = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1]
]
modelT = NN(offset)
load_npz('trained_T', modelT)
#load_npz('trained_20171108_3T', modelT)

modelS = NN(offset)
load_npz('trained_20171107S', modelS)
#load_npz('trained_20171108_3S', modelS)
#load_npz('20171110_2131_S', modelS)

modelG = NN(offset)
#load_npz('trained_G', modelG)
load_npz('trained_20171101_1G', modelG)
#load_npz('20171108_0327_G', modelG)

models = [modelT, modelS, modelG][:]
time = np.linspace(0, 1, 50).astype(np.float32).reshape(-1, 1)
color_visual = True

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

margin = 2 
color_list = ['rrrr', 'gggg', 'bbbb']

for model_i, model in enumerate(models):
    colors = color_list[model_i]
    artists = [] # animation frames
    points = []
    for u in model.u:
        W = u.link.data
        # 縮退
        W[np.where(abs(W) < 0.01)] = 0

        # 衛星のほうが小さくなるように並び替え
        argsort = np.argsort(-abs(W[0, :-2]))
        omega = u.omega.data
        phi = u.phi.data
        W[0, :-2] = W[0, argsort]
        omega[:, 0] = omega[argsort, 0]
        phi = phi[argsort]
        
        u.link = chainer.Parameter(initializer=W, shape=(1, u.dim+2))
        u.omega = chainer.Parameter(initializer=omega, shape=(u.dim, 1))
        u.phi = chainer.Parameter(initializer=phi, shape=(u.dim,))
        
        points.append( u.calc(time) )
        """plt.plot(center_x.flatten(), center_y.flatten())
        for j in range(u.dim):
            plt.plot(satellite_x[:,j], satellite_y[:,j])"""
    predict_x, predict_y = model.calc(time)
    predict_x += model_i * margin
    sys.stdout.write('.'*20)
    for i in range(len(time)):
        #if not i == 25:
        #    continue
        sys.stdout.write('\r' + '#'*(20*i//len(time)))
        im = []
        b_x = []
        b_y = []

        # for each units
        for pi, p in enumerate(points):
            base_x = p[0] + model_i * margin
            base_y = p[1]
            center_plotted = False
            for j in range(model.u[pi].dim):
                omega = abs(F.floor(model.u[pi].omega).data[j, 0])
                if not center_plotted:
                    if omega == 0:
                        base_x += p[2][i, j]
                        base_y += p[3][i, j]
                        continue
                    else:
                        if color_visual:
                            im.extend(ax.plot(base_x, base_y, '%c.'%colors[pi]))
                        else:
                            im.extend(ax.plot(base_x, base_y, color='0.3', marker='.'))
                        center_plotted = True
                
                radius = model.u[pi].link.data[0, j]
                circle_time = np.linspace(0, 1, 100)
                circle_x = radius * np.cos(circle_time * np.pi * 2)
                circle_y = radius * np.sin(circle_time * np.pi * 2)
                if color_visual:
                    im.extend(ax.plot(base_x+circle_x, base_y+circle_y, '%c-'%colors[pi], linewidth=0.5))
                else:
                    im.extend(ax.plot(base_x+circle_x, base_y+circle_y, color='0.3', linewidth=0.5))
                im.extend(ax.plot([base_x, base_x+p[2][i,j]], [base_y, base_y+p[3][i,j]], color='0.3', linewidth=0.3))
                base_x += p[2][i, j]
                base_y += p[3][i, j]
                if color_visual:
                    im.extend(ax.plot(base_x, base_y, '%c.'%colors[pi], markersize=2))
                else:
                    im.extend(ax.plot(base_x, base_y, color='0.3', marker='.', markersize=2))
            b_x.append(base_x)
            b_y.append(base_y)
        
        if color_visual:
            im.extend(ax.plot(
                [b_x[0], b_x[2], predict_x.data[i]], [b_y[0], b_y[2], predict_y.data[i]],
                '%c-'%colors[0],
                [b_x[1], b_x[3], predict_x.data[i]], [b_y[1], b_y[3], predict_y.data[i]],
                '%c-'%colors[0], linewidth=1, alpha=0.3))
            im.extend(ax.plot(predict_x.data[i], predict_y.data[i], '%co'%colors[0], alpha=0.6))
            im.extend(ax.plot(predict_x.data[:], predict_y.data[:], '%c-'%colors[0], linewidth=2))
            artists.append(im)
        
        else:
            im.extend(ax.plot(
                [b_x[0], b_x[2], predict_x.data[i]], [b_y[0], b_y[2], predict_y.data[i]],
                [b_x[1], b_x[3], predict_x.data[i]], [b_y[1], b_y[3], predict_y.data[i]], 
                '-', linewidth=0.3, color='0.3'))
            im.extend(ax.plot(predict_x.data[i], predict_y.data[i], color='0.4', marker='o', markersize=3))
            im.extend(ax.plot(predict_x.data[:], predict_y.data[:], '-', color='0', linewidth=2))
            artists.append(im)
    
    artists_three.append(artists)
    print("ok")

print("merge...")
artists = []
sys.stdout.write('.'*10)
for i in range(len(artists_three[0])):
    sys.stdout.write('\r'+'#'*(10*i//len(artists_three[0])))
    ims = []
    for group in range(len(models)):
        ims.extend(artists_three[group][i])
    artists.append(ims)
print()
#anim = ArtistAnimation(fig, artists, interval=100)
#anim.save('out_simple_bw.gif', writer='imagemagick')
plt.savefig('anime/simple_bw.svg')
plt.show()
