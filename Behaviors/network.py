class TimeResolution(Behavior) : 

    def initialize(self, net) : 
        net.dt = self.parameter("dt", 1)