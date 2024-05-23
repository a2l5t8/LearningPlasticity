from pymonntorch import *
import torch
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from Helper import plot as plotter

net = Network(behavior={1 : TimeResolution(dt = 1)})

inputLayer = NeuronGroup(net = net, size = 256, behavior={
    2 : TTFS_Encoding(input = ImagePipeline("data/bridge.tif", shape = (8, 8)), T = 40),
    # 3 : CurrentBehavior(mode = "random", pw = 3),
    # 7 : Trace(),
    
    10 : EventRecorder(['spike']) 
})

net.initialize()
net.simulate_iterations(100)

plotter.plot_spike_raster(ng)


net = Network(behavior={1 : TimeResolution(dt = 1)})

inputLayer = NeuronGroup(net = net, size = 256, behavior={
    2 : NumericalValueEncoding(input = 156.71, T = 60, sigma = 50),
    # 3 : CurrentBehavior(mode = "random", pw = 3),
    # 7 : Trace(),
    
    10 : EventRecorder(['spike']) 
})

net.initialize()
net.simulate_iterations(100)

plotter.plot_spike_raster(ng)

net = Network(behavior={1 : TimeResolution(dt = 1)})

inputLayer = NeuronGroup(net = net, size = 100, behavior={
    2 : PoissionGenerator(T = 40, lamda = 20),
    # 3 : CurrentBehavior(mode = "random", pw = 3),
    # 7 : Trace(),
    
    10 : EventRecorder(['spike']) 
})

net.initialize()
net.simulate_iterations(100)

plotter.plot_spike_raster(ng)