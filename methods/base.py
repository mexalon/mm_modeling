import numpy as np

class BaseConfig:
    def __init__(self, **kwargs):
        # model params
        self.sides = ((0, 4000), (0, 4000), (1000, 2000)) # side of the model in meters ((x from, to), (y from, to), (depth from, to))
        self.shape = (21, 21, 21)   

        # time params 
        self.time_scale = 86400 # time scale in sec.
        self.t_range = 30 # time steps 
        self.dt = 0.0001 # dt, relative to self.time_scale

        # ini pore press
        self.P0 = 0.1 # default pore Pressure is 0.1 MPa (1 bar)

        # sources
        self.sources = [
            {'loc':(2000, 2000, 1500), 'Q':np.array([1]),  'P': 1}, # P - target overpressue relative to max pressure in sources (if p0=0.1 MPa, P=-0.05 => abs pressure = 0.1 - 0.05 = 0.05 MPa)
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
        
        self.side_lenght = tuple([side[1] - side[0] for side in self.sides]) # (x length, y length, d  length) in meters 
        self.dx_dy_dz = tuple([side_l/num_p for side_l, num_p in zip(self.side_lenght, self.shape)]) # (dx, dy, dz) in meters 
        

    def __repr__(self) -> str:
        return str(self.__dict__)