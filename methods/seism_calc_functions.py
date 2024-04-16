# to event calc funcs
from scipy import stats
import numpy as np
from numpy.polynomial import polynomial as P


class Micro_Seismic_Seeds():
    def __init__(self, params, **kwargs):
        self.params = params
        self.NSAMPL = self.params.NSAMPL

        # some custom defaults
        self.azi = np.pi/4 * np.ones(self.params.shape) #  average azimuth of fractures
        self.azi_sigma = np.pi*5/180 * np.ones(self.params.shape) # its std 

        self.dip = np.pi/2 * np.ones(self.params.shape) # dip: vertical fractires
        self.dip_sigma = np.pi*5/180 * np.ones(self.params.shape) # its std 

        self.tan_phi = 0.83 * np.ones(self.params.shape) # mean value of the friction coefficient (tan phi)
        self.tan_phi_sigma = 0.025 * self.tan_phi  # its std 

        self.C = 0.1 * np.ones(self.params.shape) # cohesion, MPa
        self.C_sigma = 0.3 * self.C # its std 

        self.__dict__.update(kwargs) # updating while init with custom parameters
        self.get_seed_network() # creating seed network

   
    def get_norm_vector(self, azi, dip):
        '''
        azi, dip: np.array
        Function for converting crack azimuth and dip to normal vector
        Azimuth is considered to be the angle of the fracture direction in the XY plane (0 - 180 degrees) from the X-axis.
        Dip is the angle of dipination relative to the horizontal (0 - 90 degrees, 0 - horizontal fracture).

        Функция для перевода азимута и наклона трещины трещины в вектор нормали
        Азимутом будем считать угол направления трещины в плоскости ХY (0 - 180 градусов) от оси X,
        Наклоном - угол наклона относительно горизонтали (0 - 90 градусов, 0 - горизонтальная трещина)
        '''
        nx = np.cos(np.pi/2 + azi) * np.sin(dip)
        ny = np.sin(np.pi/2 + azi) * np.sin(dip)
        nd = np.cos(dip)
        return np.stack((nx, ny, nd), axis=-1)

    
    def get_seed_network(self):
        # Parameter distribution laws (distribution parameters - arrays with model dimensionality)
        tan_phi_distrib = stats.weibull_min(loc=self.tan_phi, scale=self.tan_phi_sigma, c=1.8) # weibull distribution с=1.8
        C_distrib = stats.weibull_min(loc=self.C, scale=self.C_sigma, c=1.8) # weibull distribution с=1.8
        azi_distrib = stats.norm(loc=self.azi, scale=self.azi_sigma) # norm distribution
        dip_distrib = stats.norm(loc=self.dip, scale=self.dip_sigma)

        # Sampling NSAMPL values
        tan_phi_rvs = tan_phi_distrib.rvs(size=(self.NSAMPL,) + self.params.shape).astype('float16')
        C_rvs = C_distrib.rvs(size=(self.NSAMPL,) + self.params.shape).astype('float16')
        azi_rvs = azi_distrib.rvs(size=(self.NSAMPL,) + self.params.shape).astype('float16')
        dip_rvs = dip_distrib.rvs(size=(self.NSAMPL,) + self.params.shape).astype('float16')

        # Transpose so that the dimension is as needed: (model sahpe, NSAMPL)
        tan_phi_rvs = np.moveaxis(tan_phi_rvs, 0, -1)
        C_rvs = np.moveaxis(C_rvs, 0, -1)
        azi_rvs = np.moveaxis(azi_rvs, 0, -1)
        dip_rvs = np.moveaxis(dip_rvs, 0, -1)

        # converting fracture orientation angles to normal vectors
        norms_rvs = self.get_norm_vector(azi_rvs, dip_rvs) 

        # updating attributes
        self.tan_phi_rvs = tan_phi_rvs
        self.C_rvs = C_rvs
        self.norms_rvs = norms_rvs

        return  tan_phi_rvs, C_rvs, norms_rvs 
    

def get_litostat_pressure(params, ro):
    # ro [g/cm3] can be either an array of the model's shape or a value.
    upper_bound_depth = params.sides[-1][0] # depth of upper bound of reservior, m
    ro_upper = np.mean(ro) # overlying rocks density = average density of model
    upper_press = upper_bound_depth * ro_upper * 1e3 * 9.81 * 1e-6 #  dh[m] * ro[kg/m3] * g [m/c2] -> [MPa]  
    
    dz = params.dx_dy_dz[-1]
    ro_arr = np.ones(params.shape) * ro * 1e3 # kg/m3
    litho_press_incr = ro_arr * dz * 9.81 * 1e-6 # [MPa] increment
    litho_press = np.cumsum(litho_press_incr, axis=2) - litho_press_incr / 2 # mean lithostatic pressure in cell
    litho_press = litho_press + upper_press # adding overlying  pressure
    return litho_press


def get_stress(params, lithostat_pressure):    
    '''
    calculates the stress tensor using Dinnik's formula
    '''
    poisson = params.poisson
    poisson_const = poisson/(1-poisson)
    point_tens = np.diag((poisson_const, poisson_const, 1)).astype('float16')
    stress_tensor = np.expand_dims((lithostat_pressure),(-2,-1)) * point_tens # (x,y,d,3,3)
    return stress_tensor


def get_norm_and_shear_stress_on_seeds(tens, norms):
    '''
    calculates normal and shear stresses on planes with orientations defined by normal vectors
    '''
    vect = (np.expand_dims(tens,-3) @ np.expand_dims(norms,-1)).squeeze(-1)
    sigma_n = (np.expand_dims(vect,-2) @ np.expand_dims(norms,-1)).squeeze(-1) # normal stresses

    vect2 = (np.expand_dims(vect, -2) @ np.expand_dims(vect, -1)).squeeze(-1) # power 2
    sigma_n2 = sigma_n**2

    tau = np.sqrt(np.abs(vect2 - sigma_n2)) # shear stresses

    sigma_n = sigma_n.squeeze(-1) # (x,y,d,NSAMPL)
    tau = tau.squeeze(-1) # (x,y,d,NSAMPL)
    return sigma_n, tau


def check_colomb_criteria(params, pore_press, tan_phi, C, sigma_n, tau):
    '''
    F = tau - (sigma_n - alpha * pore) * tg_phi - C
    '''
    pore = np.expand_dims(pore_press, -1) # it needs one more dimensions for broadcasting (x,y,d,1)
    alpha = params.alpha # Biot coefficient
    F = tau - (sigma_n - alpha * pore) * tan_phi - C
    colomb_pass = np.sum(F>0, axis=-1) # number of seeds for which the coulomb criterion was met
    return colomb_pass


def get_raw_events(params, colomb_pass):
    events_diff = np.diff(colomb_pass, axis=0).astype('float') # increment at each step in the number of seeds for which the Coulomb criterion is met (t,x,y,d)
    events_diff[events_diff<0] = 0 # positive change only 
    return events_diff


def resample_raw_events(params, raw_events):
    if np.sum(raw_events) > 0:
        sampling_factor = params.target_events_num / np.sum(raw_events) # resampling the number of events to satisfy the total number of events
        result =  raw_events * sampling_factor
    else:
        result =  raw_events # for the case when there is no events at all
        
    return result


def get_events(params, events_dens):
    frac, integ = np.modf(events_dens)
    events  = integ + (np.random.rand(*events_dens.shape)<=frac)
    return events.astype('int')


def get_GR_params(mags, nbins=100):
    '''
    mags: list of magnitudes
    a, b: G-R law parameters
    '''
    m = np.sort(mags)
    bins = np.linspace(min(m), max(m), nbins) # bins
    N = np.array([np.sum(m>=b) for b in bins])
    a, b = P.polyfit(bins, np.log10(N), deg=1)
    return a, b


def get_magnetude(size, M_min=-1, b=1):
    '''
    magnitude sampling using min magnitude and b-value
    https://doi.org/10.1007/s11069-017-2750-5
    ev - 
    '''
    return stats.expon.rvs(size=size, loc=M_min, scale=1/(b*np.log(10)))


def get_events_list(ev_matrix):
    '''
    converts events sparse matrix to [[time,x,y,d,M],...] formаt
    '''
    ev_txyd = np.array(np.nonzero(ev_matrix)).T
    ev_num = np.array([ev_matrix[*ev_txyd[ii]] for ii in range(ev_txyd.shape[0])])
    if ev_txyd.size >0:
        ev = np.repeat(ev_txyd, ev_num, axis=0)
        mag = get_magnetude(ev.shape[0], M_min=0, b=1)
        result = np.column_stack((ev, mag))
    else:
        result = np.zeros((1, ev_matrix.ndim + 1)) # for the case when there is no events at all
    return result


def pad_events(ev_list, targ_len):
    ev_len = ev_list.shape[0]
    if ev_len > targ_len:
        raise IndexError('cant pad it, data is to long')

    pad = - np.ones((targ_len - ev_len,) + ev_list.shape[1:])
    padded_ev_list = np.concatenate((ev_list, pad), axis=0)
    return padded_ev_list