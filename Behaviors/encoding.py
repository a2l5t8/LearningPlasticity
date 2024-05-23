class TTFS_Encoding(Behavior) : 
    
    def initialize(self, ng) : 

        ng.input = self.parameter("input", None)
        self.offset = self.parameter("offset", 0)
        self.T = self.parameter("T", 10)
        ng.spike = ng.vector("zeros")
        ng.spike_train = ng.vector("zeros")

    def forward(self, ng) : 

        if(ng.network.iteration - self.offset >= self.T or ng.network.iteration < self.offset) : 
            ng.spike = ng.vector("zeros")
            
        en = (255 / self.T) * (ng.network.iteration - self.offset)
        st = (255 / self.T) * (ng.network.iteration - self.offset - 1)

        ng.spike_train = ng.spike
        ng.spike = torch.logical_and(ng.input > st, ng.input <= en).byte()



class NumericalValueEncoding(Behavior) : 

    def initialize(self, ng) : 
        self.input = self.parameter("input", 5)
        self.offset = self.parameter("offset", 0)
        self.sigma = self.parameter("sigma", 2.5)
        self.T = self.parameter("T", 40)

        self.mx = self.normal(self.sigma, 1, 1)

        ng.input = ng.vector("zeros")
        ng.spike = ng.vector("zeros")

        for i in range(ng.size) : 
            ng.input[i] = self.T - int(self.normal(self.sigma, i, self.input) / self.mx * self.T)
        
        for i in range(ng.size) : 
            print(ng.input[i], end=" ")
        print()

    def forward(self, ng) : 
        ng.spike = ng.vector("zeros")
        ng.spike[ng.network.iteration == ng.input] = 1
        
    def normal(self, sigma, mu, t) :
        return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (t - mu)**2 / (2 * sigma**2))


class PoissionGenerator(Behavior) :

    def initialize(self, ng) : 
        self.offset = self.parameter("offset", [0])
        self.T = self.parameter("T", [40])
        self.lamda = self.parameter("lamda", [20])

        ng.spike_train = ng.vector("zeros")
        ng.spike = ng.vector("zeros")
        self.poisson = 0

        # ng.poisson = (1 - torch.exp(ng.input * -1 * self.lamda))

    def forward(self, ng) : 

        pat = self.pattern(ng)
        
        if(pat == -1) : 
            ng.spike_train = ng.spike
            ng.spike = torch.rand(ng.size) <= 0
        else :
            self.poisson = (np.exp(-self.lamda[pat]) * (self.lamda[pat] ** (ng.network.iteration - self.offset[pat]))) / (np.math.factorial(ng.network.iteration - self.offset[pat])) / ((np.exp(-self.lamda[pat]) * (self.lamda[pat] ** self.lamda[pat])) / (np.math.factorial(self.lamda[pat]))*2)
            ng.spike_train = ng.spike
            ng.spike = torch.rand(ng.size) <= self.poisson

    def pattern(self, ng) : 
        for i in range(len(self.offset)) : 
            if(ng.network.iteration >= self.offset[i] and ng.network.iteration - self.offset[i] <= self.T[i]) : 
                return i
        return -1

class PoissionEncoding(Behavior) :

    def initialize(self, ng) : 
        ng.input = self.parameter("input", None, required = True)
        self.offset = self.parameter("offset", 0)
        self.T = self.parameter("T", 10)
        self.lamda = self.parameter("lamda", 0.005)

        ng.spike_train = ng.vector("zeros")
        ng.spike = ng.vector("zeros")

        ng.poisson = (1 - torch.exp(ng.input * -1 * self.lamda))

    def forward(self, ng) : 
        
        if(ng.network.iteration - self.offset >= self.T or ng.network.iteration < self.offset) : 
            ng.spike_train = ng.spike
            ng.spike = torch.rand(ng.size) <= 0
        else :
            ng.spike_train = ng.spike
            ng.spike = torch.rand(ng.size) <= ng.poisson