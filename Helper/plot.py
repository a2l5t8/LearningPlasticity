def plot_V(ng, cnt = 10) : 
    plt.plot(ng['voltage', 0][:,0:cnt])
    plt.plot(ng['torch.mean(voltage)', 0], color='black')
    plt.axhline(ng['LIF_Behavior', 0].threshold, color='black', linestyle='--')


def plot_I(ng, cnt = 10) : 
    plt.plot(ng['I', 0][:,0:cnt])
    plt.xlabel('iterations')
    plt.ylabel('input current')
    plt.title('Input Current of LIF model')
    plt.show()


def plot_spike_raster(ng) :
    plt.plot(ng['spike.t', 0], ng['spike.i', 0], '.', color = "blue")
    plt.xlabel('iterations')
    plt.ylabel('neuron index')
    plt.title('Raster Plot')
    plt.show()

def plot_activity(ng) : 
    plt.plot(ng['activity', 0][:,0:1])
    plt.xlabel('iterations')
    plt.ylabel('A(t)')
    plt.title('Population activity')
    plt.show()


def plot_all(ng) : 

    plotter.plot_V(ng)
    plotter.plot_I(ng)
    plotter.plot_spike_raster(ng)
    plotter.plot_activity(ng)


def plot_cosine_similarity(syn1, syn2, syn3) :
    base = torch.cat((syn1['W', 0], syn2['W', 0], syn3['W', 0]), dim = 1)
    # baseIso = torch.cat((syn1_iso['W', 0], syn2_iso['W', 0], syn3_iso['W', 0]), dim = 1)
    A = base[:,:,0]
    B = base[:,:,1]
    # iso = baseIso[:,:,0]

    plt.plot(torch.nn.functional.cosine_similarity(A, B, dim = 1))
    # plt.plot(torch.nn.functional.cosine_similarity(B, iso, dim = 1))
    plt.title("Cosine Similarity")
    # plt.legend(["A-ISO", "B-ISO"])
    plt.show()


def plot_trace(pre, post) : 
    plt.figure(figsize = (6, 2))
    plt.plot(pre['x_trace', 0][:,])
    plt.title("Pre-synaptic trace")
    plt.show()

    plt.figure(figsize = (6, 2))
    plt.title("Post-synaptic trace")
    plt.plot(post['y_trace', 0][:,0])
    plt.show()


def plot_weight_sum(syn1, syn2) : 
    plt.plot(torch.sum(syn2['W', 0], axis = 1) + torch.sum(syn1['W', 0], axis = 1))
    plt.legend(["A", "B"])
    plt.show()


def plot_weight_specific(syn1, syn2, syn3) : 
    base = torch.cat((syn1['W', 0], syn2['W', 0], syn3['W', 0]), dim = 1)
    # baseIso = torch.cat((syn1_iso['W', 0], syn2_iso['W', 0], syn3_iso['W', 0], syn4_iso['W', 0]), dim = 1)
    A = base[:,:,0]
    B = base[:,:,1]

    plt.plot(A)
    plt.legend(["neuron{} -> A".format(i + 1) for i in range(8)] + ["isoalted"])
    plt.show()

    plt.plot(B)
    plt.legend(["neuron{} -> B".format(i + 1) for i in range(8)])
    plt.show()