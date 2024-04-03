import numpy as np

class BaseConfig:
    def __init__(self, **kwargs):
        self.side_lenght = (4000, 4000, 4000) # 4 km side  x, y, -z 
        self.shape = (21, 21, 21)   

        # time params 
        self.time_scale = 86400 # time scale in sec.
        self.t_range = 30 # time steps 
        self.dt = 0.0001 # dt, relative to self.time_scale

        # ini pore press
        self.P0 = 0.1 # initial pore Pressure is 0.1 MPa (1 bar)

        # sources
        self.sources = [
            {'loc':(2000, 2000, 2000), 'Q':np.array([1]),  'P': 1},
                        ]

        # media params:
        # fluid
        self.mu = 2 # visc cP
        self.ro = 1 # fluid dencity, g/cm3
        self.K_ro = 10**4 # MPa  dP = K_ro * (dro/ro0)  fluid compressabitity

        # solid
        self.m0 = 0.2 # porocity
        self.K_m = 10**4 # MPa  dP = K_m * (dm/m0) pore space compressabitity
        self.poisson = 0.3
        self.alpha = 1 # Biot coeff

        # seismic params
        self.NSAMPL = 100 # modeling param
        self.target_events_num = 1000 # desired number of events to distribute
    
        self.__dict__.update(kwargs) # updating while init

        self.dx_dy_dz = tuple([side_l/num_p for side_l, num_p in zip(self.side_lenght, self.shape)]) # (dx, dy, dz) in meters 

    def __repr__(self) -> str:
        return str(self.__dict__)