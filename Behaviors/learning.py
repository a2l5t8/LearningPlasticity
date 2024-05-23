class Flat_STDP(Behavior) : 

    def initialize(self, ng) :
        self.stdp_factor = self.parameter("stdp_factor", 0.015)
        self.syn_type = self.parameter("syn_type", "GLUTAMATE")
        
        ng.spike_train = ng.vector("zeros")

    def forward(self, ng) : 
        
        for syn in ng.afferent_synapses[self.syn_type] : 

            pre_post = syn.dst.spike[:, None] * syn.src.spike_train[None, :]
            post_pre = syn.dst.spike_train[:, None] * syn.src.spike[None, :]
            simu = syn.dst.spike[:, None] * syn.src.spike[None, :]


            dW_neg = self.stdp_factor * (post_pre)
            dW_pos = self.stdp_factor * (pre_post + simu)

            syn.W -= dW_neg.T
            syn.W += dW_pos.T
            syn.W = torch.clip(syn.W, 0.0, 8.0)
        
        ng.spike_train = ng.spike.clone()



class STDP(Behavior) :

    def initialize(self, ng) : 

        self.stdp_factor = self.parameter("stdp_factor", 0.015)
        self.syn_type = self.parameter("syn_type", "GLUTAMATE")
        self.test_phase = self.parameter("test_phase", 160)

    def forward(self, ng) : 

        if(ng.network.iteration >= self.test_phase) : 
            return
        
        for syn in ng.afferent_synapses[self.syn_type] : 

            post_pre = self.stdp_factor * syn.dst.y_trace * syn.src.spike[:, None]
            pre_post = self.stdp_factor * syn.src.x_trace[:,None] * syn.dst.spike
            dW =  (-post_pre + pre_post) * ng.network.dt

            syn.W = torch.clip(syn.W + dW, 0.0, 60.0)


class R_STDP(Behavior) : 

    def initialize(self, ng) : 

        ng.dp = ng.vector("zeros")
        self.syn_type = self.parameter("syn_type", "GLUTAMATE")
        self.test_phase = self.parameter("test_phase", 160)

    def forward(self, ng) : 

        if(ng.network.iteration >= self.test_phase) : 
            return

        ng.dp = ng.vector(ng.network.dop)
        
        for syn in ng.afferent_synapses[self.syn_type] : 
            
            dW = syn.elig_trace * ng.network.dop
            syn.W = torch.clip(syn.W + dW, 0.0, 60.0)

class RewardFunction(Behavior) :

    def initialize(self, ng) : 
        ng.network.dop = 0
        self.tau_d = self.parameter("tau_d", 30)

        self.reward = self.parameter("reward", 6)
        self.punish = self.parameter("punish", -2)

        self.offset = self.parameter("offset", required = True)
        self.T = self.parameter("T", required = True)
        self.target = self.parameter("target", required = True)

    def forward(self, ng) : 
        dD = -ng.network.dop/self.tau_d + self.DA(ng)
        ng.network.dop += dD * ng.network.dt

    def DA(self, ng) : 
        pat = self.pattern(ng)
        if(pat == -1) : 
            return 0
        
        if(ng.spike[ng.spike == 1].size()[0] == 1 and ng.spike[self.target[pat]] == 1) : 
            return self.reward
        elif(ng.spike[ng.spike == 1].size()[0] != 0) : 
            return self.punish
        return 0

    def pattern(self, ng) : 
        for i in range(len(self.offset)) : 
            if(ng.network.iteration >= self.offset[i] and ng.network.iteration - self.offset[i] <= self.T[i]) : 
                return i
        return -1


            