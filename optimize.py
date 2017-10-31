from scipy.optimize import minimize
import numpy as np

np.random.seed(0)
graph_t = np.r_[
    np.c_[np.linspace(48, 190, 20), np.ones(20) * 90],
    np.c_[np.ones(30) * 105, np.linspace(250, 90, 30)],
    np.c_[np.ones(30) * 105, np.linspace(90, 250, 30)],
    np.c_[np.linspace(190, 105, 10), np.ones(10) * 90],
    np.c_[np.linspace(105, 48, 10), np.ones(10) * 90]
]
time_t = np.linspace(0, 1, len(graph_t))

#periodic_time = 1.0
"""parameter = {
    'center': np.zeros((4, 2)),
    'planet_r': np.ones(4) * 10,
    'planet_omega': fixed
    'planet_phi': np.random.rand(4),
    'satellite_r': np.ones(4) * 10,
    'satellite_omega': fixed
    'satellite_phi': np.random.rand(4)
}"""

def point(time, **kwargs):
    center = kwargs['center']
    planet_phase = (np.c_[kwargs['planet_omega'] * time + kwargs['planet_phi']]) * np.pi * 2
    planet = center + np.c_[kwargs['planet_r']] * np.c_[np.cos(planet_phase), np.sin(planet_phase)]
    satellite_phase = (np.c_[kwargs['satellite_omega'] * time + kwargs['satellite_phi']]) * np.pi * 2
    satellite = planet + np.c_[kwargs['satellite_r']] * np.c_[np.cos(satellite_phase), np.sin(satellite_phase)]
    return satellite

def targetFunction(param):
    loss = 0
    for index, time in enumerate(time_t):
        points = point(time, 
            center=param[:8].reshape(4,2),
            planet_r=param[8:12], planet_omega=1, planet_phi=param[12:16],
            satellite_r=param[16:20], satellite_omega=4, satellite_phi=param[20:24])
        
        x, y = points[:, 0], points[:, 1]
        r = ((y[3]-y[1])*(x[3]-x[2]) + (x[1]-x[3])*(y[3]-y[2])) / ((x[0]-x[2])*(y[3]-y[1]) - (y[0]-y[2])*(x[3]-x[1]))
        X = r*x[0] + (1-r)*x[2]
        Y = r*y[0] + (1-r)*y[2]
        loss += np.linalg.norm(graph_t[index] - np.c_[X, Y])
    return loss

param0 = np.zeros(24)
param0[:8] = np.array([[1,0], [0,1], [-1,0], [0,-1]]).flatten()
ret = minimize(targetFunction, param0)
print(ret)