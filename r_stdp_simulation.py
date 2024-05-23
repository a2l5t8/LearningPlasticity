net = Network(behavior={1 : TimeResolution(dt = 1)})

inputLayer1 = NeuronGroup(net = net, size = 4, behavior={
    # 2 : PoissionEncoding(input = ImagePipeline("data/bird.tif", shape = (2, 2)), T = 0, lamda = 0.002),
    2 : PoissionGenerator(offset = [0, 80, 400], T = [60, 60, 60], lamda = [30, 30, 30]),
    # 3 : BackgroundActivity(),
    # 3 : CurrentBehavior(mode = "random", pw = 3),
    7 : Trace(),
    
    20 : Recorder(['x_trace','y_trace']),
    21 : EventRecorder(['spike'])
})

# inputLayerMiddle = NeuronGroup(net = net, size = 0, behavior={
#     # 2 : PoissionEncoding(input = ImagePipeline("data/bird.tif", shape = (2, 2)), T = 50, lamda = 0.002),   
#     2 : PoissionGenerator(offset = [0, 80, 160, 250, 400, 450], T = [60, 60, 60, 60, 60, 60], lamda = [30, 30, 30, 30, 30, 30]),
#     3 : BackgroundActivity(),
#     # 3 : CurrentBehavior(mode = "random", pw = 3),
#     7 : Trace(),

#     20 : Recorder(['x_trace','y_trace']),
#     21 : EventRecorder(['spike', 'bck_spike']) 
# })

inputLayer2 = NeuronGroup(net = net, size = 4, behavior={
    # 2 : PoissionEncoding(input = ImagePipeline("data/bird.tif", shape = (2, 2)), T = 50, lamda = 0.002),   
    2 : PoissionGenerator(offset = [160, 250, 450], T = [60, 60, 60], lamda = [30, 30, 30]),
    # 3 : BackgroundActivity(),
    # 3 : CurrentBehavior(mode = "random", pw = 3),
    7 : Trace(),

    20 : Recorder(['x_trace','y_trace']),
    21 : EventRecorder(['spike']) 
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
    
#     10 : Trace(),
#     12 : RewardFunction(target = [0, 0, 0, 0], offset = [0, 80, 160, 250], T = [60, 60, 60, 60], reward = 2, punish = -4),
#     13 : R_STDP(test_phase = 320),

#     20 : Recorder(['voltage', 'torch.mean(voltage)', 'I', 'activity', 'x_trace','y_trace']),
#     21 : EventRecorder(['spike', 'bck_spike']) 
# })


# --------------------------------- ISOLATED NEURONS --------------------------------------------------

outputLayer = NeuronGroup(net = net, size = 2, behavior={
    # 3 : CurrentBehavior(mode = "random", pw = 0.1),
    4 : SynTypeInput(),
    6 : LIF_Behavior(tau = 15),
    # 7 : BackgroundActivity(0.1),
    # 8 : Bias(target = 2, offset = 90, T = 10),
    # 9 : LateralInhibition(inh_rate = 180),
    9 : InitialSpike(target = [0, 1, 1, 0], offset = [0, 40, 160, 200], T = [20, 20, 20, 20]),
    
    10 : Trace(),
    12 : RewardFunction(target = [0, 0, 1, 1], offset = [0, 80, 160, 250], T = [60, 60, 60, 60], reward = 4, punish = -70),
    13 : R_STDP(test_phase = 320),

    20 : Recorder(['voltage', 'torch.mean(voltage)', 'I', 'activity', 'x_trace','y_trace', 'dp']),
    21 : EventRecorder(['spike']) 
})


syn1 = SynapseGroup(net = net, src = inputLayer1, dst = outputLayer, tag = "GLUTAMATE", behavior={
    2 : SynConnectivity(mode = "full2", J0 = 10),
    3 : SynFun(),
    
    11 : EligTrace(),

    20 : Recorder(['W', 'elig_trace']),
})

syn2 = SynapseGroup(net = net, src = inputLayer2, dst = outputLayer, tag = "GLUTAMATE", behavior={
    2 : SynConnectivity(mode = "full2", J0 = 10),
    3 : SynFun(),
    
    11 : EligTrace(),

    20 : Recorder(['W', 'elig_trace']),
})

# syn3 = SynapseGroup(net = net, src = inputLayerMiddle, dst = outputLayer, tag = "GLUTAMATE", behavior={
#     2 : SynConnectivity(mode = "full2", J0 = 10),
#     3 : SynFun(),

#     11 : EligTrace(),

#     20 : Recorder(['W', 'elig_trace']),
# })

# --------------------------------- ISOLATED NEURONS --------------------------------------------------

# syn1_iso = SynapseGroup(net = net, src = inputLayer1, dst = outputLayerIsolated, tag = "GLUTAMATE", behavior={
#     2 : SynConnectivity(mode = "full2", J0 = 2),
#     3 : SynFun(),

#     11 : EligTrace(),

#     20 : Recorder(['W']),
# })

# syn2_iso = SynapseGroup(net = net, src = inputLayer2, dst = outputLayerIsolated, tag = "GLUTAMATE", behavior={
#     2 : SynConnectivity(mode = "full2", J0 = 2),
#     3 : SynFun(),

#     11 : EligTrace(),

#     20 : Recorder(['W']),
# })

# syn3_iso = SynapseGroup(net = net, src = inputLayerMiddle, dst = outputLayerIsolated, tag = "GLUTAMATE", behavior={
#     2 : SynConnectivity(mode = "full2", J0 = 2),
#     3 : SynFun(),

#     11 : EligTrace(),

#     20 : Recorder(['W']),
# })

# syn4_iso = SynapseGroup(net = net, src = inputLayerIsolated, dst = outputLayerIsolated, tag = "GLUTAMATE", behavior={
#     2 : SynConnectivity(mode = "full2", J0 = 2),
#     3 : SynFun(),

#     11 : EligTrace(),

#     20 : Recorder(['W']),
# })

# syn5_iso = SynapseGroup(net = net, src = inputLayerIsolated, dst = outputLayer, tag = "GLUTAMATE", behavior={
#     2 : SynConnectivity(mode = "full2", J0 = 2),
#     3 : SynFun(),

#     11 : EligTrace(),

#     20 : Recorder(['W']),
# })

# --------------------------------- ISOLATED NEURONS --------------------------------------------------

net.initialize()
net.simulate_iterations(500)


plt.figure(figsize = (6, 2))
plt.plot(inputLayer1['spike.t', 0], inputLayer1['spike.i', 0], '.', color = "blue")
# plt.plot(inputLayerMiddle['spike.t', 0], inputLayerMiddle['spike.i', 0] + 2, '.', color = "purple")
plt.plot(inputLayer2['spike.t', 0], inputLayer2['spike.i', 0] + 4, '.', color = "orange")
# plt.plot(inputLayerIsolated['spike.t', 0] + 8, inputLayerIsolated['spike.i', 0] + 8, '.', color = "green")

# plt.plot(inputLayer1['bck_spike.t', 0], inputLayer1['bck_spike.i', 0], '.', color = "red")
# plt.plot(inputLayerMiddle['spike.t', 0], inputLayerMiddle['spike.i', 0] + 3, '.', color = "purple")
# plt.plot(inputLayer2['bck_spike.t', 0], inputLayer2['bck_spike.i', 0] + 4, '.', color = "red")
# plt.plot(inputLayerIsolated['bck_spike.t', 0] + 4, inputLayerIsolated['bck_spike.i', 0] + 8, '.', color = "red")

plt.legend(["inp1", "inp2"])
plt.title("Input layer raster plot")

plt.figure(figsize = (6, 2))
plt.plot(outputLayer['spike.t', 0], outputLayer['spike.i', 0], '.', color = "blue", marker = "|", markersize = 20, markeredgewidth=1.4)
plt.plot([350, 350], [-0.1, 1.1], color = "black")
# plt.plot(outputLayer['bck_sNpike.t', 0], outputLayer['bck_spike.i', 0], '.', color = "orange", marker = "|", markersize = 20, markeredgewidth=1.4)
# plt.plot(outputLayerIsolated['spike.t', 0] + 2, outputLayerIsolated['spike.i', 0] + 2, '.', color = "orange", marker = "|", markersize = 20, markeredgewidth=1.4)
plt.plot([350, 350], [-0.1, 1.1], color = "black")

plt.legend(["A, B","Testing phase"])
plt.title("Output layer raster plot")

plt.plot(net["dp", 0][:,0])
plt.title("Dopamine Rate")

base = torch.cat((syn1['W', 0], syn2['W', 0], syn3['W', 0]), dim = 1)
# baseIso = torch.cat((syn1_iso['W', 0], syn2_iso['W', 0], syn3_iso['W', 0], syn4_iso['W', 0]), dim = 1)
A = base[:,:,0]
B = base[:,:,1]

# iso = baseIso[:,:,0]

plt.plot(torch.nn.functional.cosine_similarity(A, B, dim = 1))
# plt.plot(torch.nn.functional.cosine_similarity(B, iso, dim = 1))
plt.title("Cosine Similarity")
# plt.legend(["A-ISO", "B-ISO"])

plt.plot(A)
plt.legend(["neuron{} -> A".format(i + 1) for i in range(8)] + ["isoalted"])
plt.show()

plt.plot(B)
plt.legend(["neuron{} -> B".format(i + 1) for i in range(8)])
plt.show()