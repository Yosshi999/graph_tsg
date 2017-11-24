import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from scipy import interpolate
import sys
"""
    TODO: path to array
"""

class unit(chainer.Chain):
    def __init__(self, x, y):
        super().__init__()
        self.dim = 4
        self.offset = chainer.Variable(np.hstack((np.zeros((1, self.dim)), np.array([[x, 0]]))).astype(np.float32))
        
        dim = self.dim
        with self.init_scope():
            self.omega = chainer.Parameter(
                initializer=np.random.rand(dim, 1).astype(np.float32) * 10 - 5,
                shape=(dim, 1))
            self.phi = chainer.Parameter(
                initializer=np.random.rand(dim).astype(np.float32),
                shape=(dim,))
            linkinit = np.random.rand(1, dim+2).astype(np.float32) * 0.5
            #linkinit = np.zeros((1, dim+2)).astype(np.float32)
            linkinit[0, -2] = x
            linkinit[0, -1] = y
            self.link = chainer.Parameter(initializer=linkinit, shape=(1, dim+2))
        #self.omega = chainer.Variable(np.array([[1.], [4.]], np.float32))
    def __call__(self, time):
        _0 = chainer.Variable(np.zeros((len(time),1), np.float32))
        _1 = chainer.Variable(np.ones((len(time),1), np.float32))

        phase = F.linear(np.c_[time], F.floor(self.omega), self.phi)*2*np.pi
        x = F.cos(phase)
        y = F.sin(phase)
        x = F.linear(F.concat((x, _1, _0)), self.link)
        y = F.linear(F.concat((y, _0, _1)), self.link)
        self.l1 = F.sum(F.absolute(F.concat((_1,)*self.dim + (_0, _0)) * F.vstack((self.link,)*len(time))))
        
        base = chainer.Variable(np.array([[0]*self.dim + [1, 0]], np.float32)) * self.link
        self.off_loss = F.sum((base - self.offset)**2)
        return x, y
    def calc(self, time):
        _0 = chainer.Variable(np.zeros((len(time),1), np.float32))
        _1 = chainer.Variable(np.ones((len(time),1), np.float32))

        phase = F.linear(np.c_[time], F.floor(self.omega), self.phi)*2*np.pi
        x = F.cos(phase)
        y = F.sin(phase)
        center_x = F.linear(F.concat((*[_0 for _ in range(self.dim)] , _1, _0)), self.link).data
        center_y = F.linear(F.concat((*[_0 for _ in range(self.dim)], _0, _1)), self.link).data
        satellite_x = x.data * self.link.data[:,:4]
        satellite_y = y.data * self.link.data[:, :4]
        return center_x[0,0], center_y[0,0], satellite_x, satellite_y

class NN(chainer.Chain):
    def __init__(self, offset):
        super().__init__()
        self.train = True
        with self.init_scope():
            self.u0 = unit(*offset[0])
            self.u1 = unit(*offset[1])
            self.u2 = unit(*offset[2])
            self.u3 = unit(*offset[3])
        self.u = [self.u0, self.u1, self.u2, self.u3]

    def calc(self, time):
        x = [None for _ in range(4)]
        y = [None for _ in range(4)]
        for i in range(4):
            x[i], y[i] = self.u[i](time)
        r = ((y[3]-y[1])*(x[3]-x[2]) + (x[1]-x[3])*(y[3]-y[2])) / ((x[0]-x[2])*(y[3]-y[1]) - (y[0]-y[2])*(x[3]-x[1]))
        X = r*x[0] + (1-r)*x[2]
        Y = r*y[0] + (1-r)*y[2]
        return X, Y

    def __call__(self, time, teacher, sigma):
        """return loss (L2 distance + L1 norm)"""
        l1_lambda = 0.001
        teacher_x, teacher_y = np.c_[teacher[...,0]], np.c_[teacher[...,1]]

        predict_x, predict_y = self.calc(time)
        diff_vec = F.concat((predict_x-teacher_x, predict_y-teacher_y))
        
        loss = F.sum(diff_vec**2)
        
        l1 = sum([u.l1 for u in self.u])
        off_loss = sum([u.off_loss for u in self.u])
        loss += l1 * l1_lambda
        loss += off_loss * 0.1
        #loss = F.sum((teacher_x-predict_x)**2 + (teacher_y-predict_y)**2)
        if self.train:
            chainer.reporter.report({'main/loss': loss.data.sum()/len(time)})
        else:
            chainer.reporter.report({'validation/main/loss': loss.data.sum()/len(time)})
            chainer.reporter.report({'validation/main/L1': l1.data.sum()/len(time)})
        return loss

class TestEvaluator(extensions.Evaluator):
    def __init__(self, test_iter, model, trainer):
        super(TestEvaluator, self).__init__(test_iter, model)
        self.trainer = trainer
    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestEvaluator, self).evaluate()
        model.train = True
        return ret
 

if __name__ == '__main__':
    name = sys.argv[1]
    prefix = sys.argv[2]
    train_from = 0
    train_to = 60
    train = True
    load = sys.argv[3] if sys.argv[3] != 'None' else None
    epoch = int(sys.argv[4])

    offset = None
    if name == 'T':
        offset = [ # T
            [-3.5, -3.5],
            [-2, -2],
            [-2, 2],
            [-3.5, 3.5]
        ]
    elif name == 'S':
        offset = [ # T
            [-1, -2],
            [1, -2],
            [1, 2],
            [-1, 2]
        ]
    elif name == 'G':
        offset = [ # G
            [1.2, -1.2], # 左上
            [2.5, -2.5], # 右上
            [2.5, 2.5], # 右下
            [1.2, 1.2]  # 左下
        ]

    np.random.seed(132)
    model = NN(offset)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    save_name = '%s%s'%(prefix, name)
    if load:
        chainer.serializers.load_npz(load, model)
    """for i in range(4):
        W = model.u[i].link.data
        W[0, -2] = offset[i][0]
        W[0, -1] = offset[i][1]
        model.u[i].link = chainer.Parameter(initializer=W, shape=(1, model.u[i].dim+2))
    """
    # T
    if name == 'T':
        _t = np.array([
            [30, 71],
            [168, 69],
            [96, 77],
            [98, 185]
        ])
        _t = np.vstack((_t, _t[::-1]))
        graph_tx = interpolate.interp1d(np.linspace(0, 1, len(_t)), _t[:,0])(np.linspace(0, 1, 60)).reshape(-1,1)
        graph_ty = interpolate.interp1d(np.linspace(0, 1, len(_t)), _t[:,1])(np.linspace(0, 1, 60)).reshape(-1,1)
        graph_t = np.hstack((graph_tx, graph_ty)).astype(np.float32)


    # S
    if name == 'S':
        _s = np.array([
            [181, 66],
            [164, 55],
            [141, 48],
            [116, 42],
            [95, 59],
            [85, 80],
            [82, 103],
            [98, 122],
            [115, 135],
            [136, 147],
            [155, 155],
            [170, 174],
            [173, 200],
            [145, 215],
            [125, 223],
            [102, 219],
            [85, 206],
            [72, 189]
        ])
        _s = np.vstack((_s, _s[::-1]))
        graph_tx = interpolate.interp1d(np.linspace(0, 1, len(_s)), _s[:,0])(np.linspace(0, 1, 60)).reshape(-1,1)
        graph_ty = interpolate.interp1d(np.linspace(0, 1, len(_s)), _s[:,1])(np.linspace(0, 1, 60)).reshape(-1,1)
        graph_t = np.hstack((graph_tx, graph_ty)).astype(np.float32)



    # G
    if name == 'G':
        _g = np.array([
            [163, 73],
            [149, 51],
            [122, 38],
            [86, 42],
            [69, 56],
            [45, 77],
            [33, 108],
            [33, 133],
            [38, 165],
            [50, 187],
            [70, 202],
            [100, 214],
            [133, 204],
            [158, 189],
            [172, 166],
            [174, 139],
            [150, 138],
            [121, 140],
            [121, 140]
        ])
        _g = np.vstack((_g, _g[::-1]))
        graph_tx = interpolate.interp1d(np.linspace(0, 1, len(_g)), _g[:,0])(np.linspace(0, 1, 60)).reshape(-1,1)
        graph_ty = interpolate.interp1d(np.linspace(0, 1, len(_g)), _g[:,1])(np.linspace(0, 1, 60)).reshape(-1,1)
        graph_t = np.hstack((graph_tx, graph_ty)).astype(np.float32)

    size = graph_t.max() - graph_t.min()
    graph_t -= graph_t.min() + size/2
    graph_t /= size

    """plt.scatter(graph_t[:,0], graph_t[:,1], c="red")
    plt.show()
    raise"""

    dxdt = np.vstack((graph_t[1:], graph_t[0])) - graph_t # 使わない
    sigma = (dxdt + dxdt[:,::-1]/30)*10 + 1 # 使わない

    time_iterator = np.linspace(0, 1, len(graph_t)).astype(np.float32).reshape(-1, 1)

    """times = np.linspace(0, 1, 10).reshape(10, 1).astype(np.float32)
    x, y = model.calc(times)
    plt.scatter(x.data, y.data, c="blue")
    plt.show()
    raise"""


    train_dataset = chainer.datasets.TupleDataset(
        time_iterator[train_from:train_to],
        graph_t[train_from:train_to],
        sigma[train_from:train_to])

    train_iter = chainer.iterators.SerialIterator(train_dataset, 1, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(train_dataset, train_to-train_from, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer)

    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'))
    #trainer.extend(extensions.snapshot(), trigger=(50, 'epoch'))
    trainer.extend(TestEvaluator(test_iter, model, trainer))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'validation/main/L1', 'elapsed_time']
    ))

    if train:
        trainer.run()

    for u_i, u in enumerate(model.u):
        print('unit %d:'%u_i, u.link.data, end=' ')
        print(len(np.where(abs(u.link.data) > 0.1)[0]))

    times = np.linspace(0, 1, len(graph_t)).reshape(-1, 1).astype(np.float32)
    predict_x, predict_y = model.calc(times)
    teacher = graph_t
    plt.figure(figsize=(8,4))
    
    plt.subplot(1,2,1)
    plt.scatter(teacher[train_from:train_to, 0], teacher[train_from:train_to, 1], c="red")
    plt.plot(predict_x.data[train_from:train_to], predict_y.data[train_from:train_to], "b.-")
    plt.xlim(-1, 1)
    plt.ylim(1, -1)
    #plt.show()

    plt.subplot(1,2,2)
    plt.scatter(teacher[:, 0], teacher[:, 1], c="0.3")
    plt.plot(predict_x.data[:], predict_y.data[:], "-", c="0", linewidth=2)
    plt.xlim(-1, 1)
    plt.ylim(1, -1)
    plt.show()

    print('save? >>', end='')
    if train and input() == 'y':
        chainer.serializers.save_npz(save_name, model)
        print('saved. (%s)'%save_name)