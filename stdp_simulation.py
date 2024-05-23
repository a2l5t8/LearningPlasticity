net = Network(behavior={1 : TimeResolution(dt = 1)})

inputLayer1 = NeuronGroup(net = net, size = 4, behavior={
    # 2 : PoissionEncoding(input = ImagePipeline("data/bird.tif", shape = (2, 2)), T = 0, lamda = 0.002),
    2 : PoissionGenerator(offset = [0, 160], T = [60, 60], lamda = [30, 30]),
    3 : BackgroundActivity(),
    # 3 : CurrentBehavior(mode = "random", pw = 3),
    7 : Trace(),
    
    20 : Recorder(['x_trace','y_trace']),
    21 : EventRecorder(['spike', 'bck_spike'])
})

inputLayerMiddle = NeuronGroup(net = net, size = 0, behavior={
    # 2 : PoissionEncoding(input = ImagePipeline("data/bird.tif", shape = (2, 2)), T = 50, lamda = 0.002),   
    2 : PoissionGenerator(offset = [0, 60], T = [40, 40], lamda = [20, 20]),
    3 : BackgroundActivity(),
    # 3 : CurrentBehavior(mode = "random", pw = 3),
    7 : Trace(),

    20 : Recorder(['x_trace','y_trace']),
    21 : EventRecorder(['spike', 'bck_spike']) 
})

inputLayer2 = NeuronGroup(net = net, size = 4, behavior={
    # 2 : PoissionEncoding(input = ImagePipeline("data/bird.tif", shape = (2, 2)), T = 50, lamda = 0.002),   
    2 : PoissionGenerator(offset = [80, 250], T = [60, 60], lamda = [30, 30]),
    3 : BackgroundActivity(),
    # 3 : CurrentBehavior(mode = "random", pw = 3),
    7 : Trace(),

    20 : Recorder(['x_trace','y_trace']),
    21 : EventRecorder(['spike', 'bck_spike']) 
})


# --------------------------------- ISOLATED NEURONS --------------------------------------------------

# inputLayerIsolated = NeuronGroup(net = net, size = 1, behavior={
#     # 2 : PoissionEncoding(input = ImagePipeline("data/bird.tif", shape = (2, 2)), T = 50, lamda = 0.002),   
#     2 : PoissionGenerator(offset = [0], T = [0], lamda = [0]),
#     3 : BackgroundActivity(),
#     # 3 : CurrentBehavior(mode = "random", pw = 3),
#     7 : Trace(),

#     9 : Recorder(['x_trace','y_trace']),
#     10 : EventRecorder(['spike', 'bck_spike']) 
# })

# outputLayerIsolated = NeuronGroup(net = net, size = 1, behavior={
#     # 3 : CurrentBehavior(mode = "random", pw = 0.1),
#     4 : SynTypeInput(),
#     6 : LIF_Behavior(tau = 15, threshold = 10000),
#     7 : BackgroundActivity(0.1),
    
#     8 : Trace(),
#     9 : STDP(stdp_factor = 0.2),

#     10 : Recorder(['voltage', 'torch.mean(voltage)', 'I', 'activity', 'x_trace','y_trace']),
#     11 : EventRecorder(['spike', 'bck_spike']) 
# })


# --------------------------------- ISOLATED NEURONS --------------------------------------------------

outputLayer = NeuronGroup(net = net, size = 2, behavior={
    # 3 : CurrentBehavior(mode = "random", pw = 0.1),
    4 : SynTypeInput(),
    6 : LIF_Behavior(tau = 10),
    7 : BackgroundActivity(0.1),
    8 : Bias(target = 2, offset = 90, T = 10),
    9 : LateralInhibition(inh_rate = 180),
    
    10 : Trace(),
    11 : STDP(stdp_factor = 2),

    20 : Recorder(['voltage', 'torch.mean(voltage)', 'I', 'activity', 'x_trace','y_trace']),
    21 : EventRecorder(['spike', 'bck_spike']) 
})


syn1 = SynapseGroup(net = net, src = inputLayer1, dst = outputLayer, tag = "GLUTAMATE", behavior={
    2 : SynConnectivity(mode = "full2", J0 = 10),
    3 : SynFun(),
    9 : Recorder(['W']),
})

syn2 = SynapseGroup(net = net, src = inputLayer2, dst = outputLayer, tag = "GLUTAMATE", behavior={
    2 : SynConnectivity(mode = "full2", J0 = 10),
    3 : SynFun(),
    9 : Recorder(['W']),
})

syn3 = SynapseGroup(net = net, src = inputLayerMiddle, dst = outputLayer, tag = "GLUTAMATE", behavior={
    2 : SynConnectivity(mode = "full2", J0 = 10),
    3 : SynFun(),
    9 : Recorder(['W']),
})

# --------------------------------- ISOLATED NEURONS --------------------------------------------------

# syn1_iso = SynapseGroup(net = net, src = inputLayer1, dst = outputLayerIsolated, tag = "GLUTAMATE", behavior={
#     2 : SynConnectivity(mode = "full2", J0 = 2),
#     3 : SynFun(),
#     9 : Recorder(['W']),
# })

# syn2_iso = SynapseGroup(net = net, src = inputLayer2, dst = outputLayerIsolated, tag = "GLUTAMATE", behavior={
#     2 : SynConnectivity(mode = "full2", J0 = 2),
#     3 : SynFun()zZ,
#     9 : Recorder(['W']),
# })

# syn3_iso = SynapseGroup(net = net, src = inputLayerMiddle, dst = outputLayerIsolated, tag = "GLUTAMATE", behavior={
#     2 : SynConnectivity(mode = "full2", J0 = 2),
#     3 : SynFun(),
#     9 : Recorder(['W']),
# })

# --------------------------------- ISOLATED NEURONS --------------------------------------------------

net.initialize()
net.simulate_iterations(320)




plt.figure(figsize = (6, 2))
plt.plot(inputLayer1['spike.t', 0], inputLayer1['spike.i', 0], '.', color = "blue")
# plt.plot(inputLayerMiddle['spike.t', 0], inputLayerMiddle['spike.i', 0] + 3, '.', color = "purple")
plt.plot(inputLayer2['spike.t', 0], inputLayer2['spike.i', 0] + 4, '.', color = "orange")
# plt.plot(inputLayerIsolated['spike.t', 0] + 4, inputLayerIsolated['spike.i', 0] + 8, '.', color = "green")

plt.plot(inputLayer1['bck_spike.t', 0], inputLayer1['bck_spike.i', 0], '.', color = "red")
# plt.plot(inputLayerMiddle['spike.t', 0], inputLayerMiddle['spike.i', 0] + 3, '.', color = "purple")
plt.plot(inputLayer2['bck_spike.t', 0], inputLayer2['bck_spike.i', 0] + 4, '.', color = "red")
# plt.plot(inputLayerIsolated['bck_spike.t', 0] + 4, inputLayerIsolated['bck_spike.i', 0] + 8, '.', color = "red")

plt.legend(["inp1","inp2", "bck"])


plt.figure(figsize = (6, 2))
plt.plot(outputLayer['spike.t', 0], outputLayer['spike.i', 0], '.', color = "blue", marker = "|", markersize = 20, markeredgewidth=1.4)
plt.plot(outputLayer['bck_spike.t', 0], outputLayer['bck_spike.i', 0], '.', color = "orange", marker = "|", markersize = 20, markeredgewidth=1.4)
# plt.plot(outputLayerIsolated['spike.t', 0] + 2, outputLayerIsolated['spike.i', 0] + 2, '.', color = "orange", marker = "|", markersize = 20, markeredgewidth=1.4)
plt.legend(["A, B", "background"])