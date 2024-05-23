class LateralInhibition(Behavior) : 

    def initialize(self, ng) : 
        self.inh_rate = self.parameter("inh_rate", 3)

    def forward(self, ng) : 
        
        inhibition = ng.vector(self.inh_rate)
        if(inhibition[ng.spike == 1].size()[0] != 0) :
            inhibition[ng.spike == 1] = 0
        else :
            inhibition = ng.vector(0)
            
        ng.I -= inhibition

class Trace(Behavior) : 

    def initialize(self, ng) : 
        self.trace_tau = self.parameter("trace_tau", 10)
        
        ng.x_trace = ng.vector("zeros")
        ng.y_trace = ng.vector("zeros")

    def forward(self, ng) :
        
        dX = (-1) * ng.x_trace/self.trace_tau + ng.spike
        ng.x_trace += dX * ng.network.dt

        dY = (-1) * ng.y_trace/self.trace_tau + ng.spike
        ng.y_trace += dY * ng.network.dt

class Bias(Behavior) : 
    
    def initialize(self, ng) : 
        self.offset = self.parameter("offset", 100)
        self.T = self.parameter("T", 10)
        self.bias = self.parameter("bias", 0.5)

        self.target = self.parameter("target", 2) - 1

    def forward(self, ng) : 
        
        if(ng.network.iteration < self.offset or ng.network.iteration > self.offset + self.T) : 
            return

        prob = random.random()
        if((prob <= self.bias)) :
            ng.spike[self.target] = 1

class BackgroundActivity(Behavior) : 

    def initialize(self, ng) : 
        self.rate = self.parameter("rate", 0.005)
        ng.bck_spike = ng.vector("zeros")

    def forward(self, ng) : 
        a = torch.rand(ng.size)
        ng.bck_spike = a <= self.rate
        ng.spike[a <= self.rate] = 1


class EligTrace(Behavior) : 

    def initialize(self, sg) : 
        
        self.tau_c = self.parameter("tau_c", 30)
        self.factor = self.parameter("factor", 0.005)
        sg.elig_trace = sg.matrix("zeros")

    def forward(self, sg) : 
        
        dTrace = -sg.elig_trace/self.tau_c + self.deltaW(sg)
        sg.elig_trace += dTrace * sg.network.dt

    def deltaW(self, sg) : 

        post_pre = self.factor * sg.dst.y_trace * sg.src.spike[:, None]
        pre_post = self.factor * sg.src.x_trace[:,None] * sg.dst.spike

        dW =  (-post_pre + pre_post) * sg.network.dt
        return dW

class InitialSpike(Behavior) : 

    def initialize(self, ng) : 
        
        self.offset = self.parameter("offset", required = True)
        self.T = self.parameter("T", required = True)
        self.target = self.parameter("target", required = True)
        self.rate = self.parameter("rate", 0.4)
    
    def forward(self, ng) : 
        pat = self.pattern(ng)
        if(pat == -1) : 
            return

        ng.spike[self.target[pat]] |= random.random() <= self.rate

    def pattern(self, ng) : 
        for i in range(len(self.offset)) : 
            if(ng.network.iteration >= self.offset[i] and ng.network.iteration - self.offset[i] <= self.T[i]) : 
                return i
        return -1