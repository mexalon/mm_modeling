class BaseConfig:
    def __init__(self, **kwargs):
        self.side_lenght = (4000, 4000, 4000) # 4 km side  x, y, -z 
        self.shape = (20, 20, 20)   

        # time params 
        self.time_scale = 86400 # time scale in sec.
        self.t_range = 30 # time steps 
        self.dt = 0.0001 # dt, relative to self.time_scale

        # ini pore press
        self.P0 = 10e-8 # initial pore Pressure is zero MPa

        # sources
        self.sources = [
            {'loc':(2000, 2000, 2000), 'Q':[1]},
                        ]

        # self.Q = [1] # m^3/s
        # self.source_loc = (2000, 2000, 2000) # meter x, y, -z

        # media params
        self.m0 = 0.2 # porocity
        self.mu = 2 # visc cP
        self.K_ro = 10**4 # MPa  dP = K_ro * (dro/ro0)  
        self.K_m = 10**4 # MPa  dP = K_m * (dm/m0) 

        self.__dict__.update(kwargs) # updating while init

        self.dx_dy_dz = tuple([side_l/num_p for side_l, num_p in zip(self.side_lenght, self.shape)]) # (dx, dy, dz) in meters 

    def __repr__(self) -> str:
        return str(self.__dict__)