import numpy as np
import h5py
from tqdm.notebook import tqdm
from scipy.interpolate import make_interp_spline
from pde import PDEBase, ScalarField, VectorField, MemoryStorage, CartesianGrid
from pde.tools.numba import jit


class Pore_Press_Diffusion_with_Q_source(PDEBase):
    """Diffusion equation with spatial dependence"""
    def __init__(self, perm, params):
        super().__init__()
        cache_rhs = True
        # explicit_time_dependence = True
        self.perm = perm
        self.params = params
        self.shape = self.perm.shape
        self.source_locs = [l['loc'] for l in params.sources]  # list of source locations
        self.source_time_interpolators = [self.make_interp_from_q_list(l['Q']) for l in params.sources] # list of source time funcs

        # generate grid
        x_length, y_length, z_length = params.side_lenght 
        self.grid = CartesianGrid([[0, x_length], [0, y_length], [0, z_length]], self.shape)  

        # boundary condition
        # self.bc = {"curvature": 0} # zero 2nd derivative at all boundaries
        self.bc = {"derivative": 0} # Neumann

        # diffusion coefficient scalar field
        self.xi_field = self.get_xi_field()

        # initial pore pressure
        self.pore_ini_field = self.get_pore_ini_field()

        # initial zero source field
        self.source_field = self.update_source_field(0) # spatial source field, initial state

   
    def make_interp_from_q_list(self, q):
        t = np.linspace(0, self.params.t_range, q.size)
        f = make_interp_spline(t, q, k=1)
        return f
    

    def get_xi(self):
        '''
        perm - permeability np.array 
    
        Function for calculating the diffusion coefficient. Details - Barenblatt p. 19
        Dimensionality:
        1 MPa = 10^7 g/cm*s^2
        1 mD = 10^-11 cm^2
        1 cP = 10^-2 g/cm*s
        Therefore, the dimensionality of [xi] is:
        [xi] = [mD] * [MPa] / [cP] =
            = 10^-11 [cm^2] * 10^7 [g/cms^2] / 10^-2 [g/cms] =
            = 10^-2 [cm^2/s] =
            = 10^-6 [m^2/s]

        returns numpy array with distributed xi value    
        '''
        xi = 10**-6 * self.perm / (self.params.m0 * self.params.mu * (1/self.params.K_ro + 1/self.params.K_m)) # m^2/s
        return xi * self.params.time_scale # rescaling to time scale
    
    
    def get_xi_field(self):
        '''
        numpy xi --> scalar field obj xi
        '''
        xi = self.get_xi() # xi
        return ScalarField(self.grid, data=xi)
    

    def update_source_field(self, t):
        '''
        assembling source points and applying specific rates. Iterates over source locations.
        '''
        source_field = ScalarField(self.grid, data=0) # spatial source field
        for loc, iter_q in zip(self.source_locs, self.source_time_interpolators):
            q = iter_q(t).item()
            dpdt = self.dPdt_Q(q)
            source_field.insert(loc , dpdt)  #  one borehole == point source at params.source_loc location
            
        self.source_field = source_field
        return source_field
    
    
    def dPdt_Q(self, Q):
        '''
        arg  - some float value of Q = dv/dt
        dP/dt ~ K_m * ((dv/dt)/V) - pressure change when dv of fluid is pumped in volume V
        returns float source value, rescaled to time scale
        '''
        V = 2000 * 3.1415 * 0.1**2 / 4  # some "volume" of source (borehole). m^3
        dP = self.params.K_ro * Q / V # pressure change MPa/s
        return dP * self.params.time_scale # rescaling to time scale
    
    
    def get_pore_ini_field(self):
        '''
        method to ubtain pore pressure initial state. Just for convenience here.
        '''
        p0 = np.ones_like(self.perm) * self.params.P0
        pore_pressure_field = ScalarField(self.grid, data=p0)
        return pore_pressure_field
    
    
    def _make_pde_rhs_numba(self, state):
        """ 
        the numba-acceleratin
        it freezes all values when compiling the function, 
        so the diffusivity cannot be altered without recompiling.
        Have no idea how to use it with time-dependet parameters.
        Now it works with initial value of source field 

        just uncomment this method to use
        """
        # make attributes locally available
        xi_field = self.xi_field.data
        source_field = self.source_field.data

        # create operators
        laplace = state.grid.make_operator("laplace", bc=self.bc)
        gradient = state.grid.make_operator("gradient", bc=self.bc)
        gradient_xi = state.grid.make_operator("gradient", bc="derivative")
        dot = VectorField(state.grid).make_dot_operator()

        @jit
        def pde_rhs(state_data, t=0):
            """ compiled helper function evaluating right hand side """
            lapace_P = laplace(state_data)
            grad_P = gradient(state_data)
            grad_xi = gradient_xi(xi_field)
            dP_dt = xi_field * lapace_P + dot(grad_xi, grad_P) + source_field
            return dP_dt

        return pde_rhs
    

    def evolution_rate(self, state, t=0):
        ''' all magic here '''
        # self.update_source_field(t) # updating source field. implicit source time dependence here
        grad_xi = self.xi_field.gradient(bc="derivative")
        lapace_P = state.laplace(bc=self.bc)
        grad_P = state.gradient(bc=self.bc)
        dP_dt = self.xi_field * lapace_P + grad_xi @ grad_P + self.source_field
        return dP_dt


def test_wells(params):
    """testing well injectivity"""
    for s in params.sources:
        s['Q'] = np.ones((2,)) * s['Q'][np.argmax(np.abs(s['Q'] - params.P0))] # setting q for testing - max overpressure value

    k_list = 10**np.linspace(-1, 4, 20) # list of perms to test. it takes about 1 hr
    locs = np.array([l['loc'] for l in params.sources])
    pore_press_in_locs = np.zeros((locs.shape[0], len(k_list)))
    eq = Pore_Press_Diffusion_with_Q_source(np.ones(params.shape), params) # for super fast calc
    p0 = eq.pore_ini_field
    for idx, kk in enumerate(tqdm(k_list)):
        eq.perm = kk * np.ones(params.shape) # updating perm
        eq.xi_field = eq.get_xi_field()
        storage = MemoryStorage()
        final_pore_field = eq.solve(p0, t_range=params.t_range, dt=params.dt, solver='scipy', tracker=storage.tracker(1))
        pf = final_pore_field.make_interpolator()
        pore_press_in_locs[:,idx] = pf(locs)
    
    interpolators = [make_interp_spline(k_list, press, k=1) for press in pore_press_in_locs] # creating interpolatorc p(k) 

    with h5py.File('model_press_q_testing.h5', 'w') as targ: # saving 
        targ.create_dataset("p_k", data=pore_press_in_locs)
        targ.create_dataset("k_list", data=k_list)
    
    return pore_press_in_locs, interpolators

def get_perms_at_locs(perm, params):
    """retrieving perms at loc coordinates"""
    # generate grid
    x_length, y_length, z_length = params.side_lenght 
    grid = CartesianGrid([[0, x_length], [0, y_length], [0, z_length]], params.shape)
    perm_field = ScalarField(grid, data=perm)
    perms_at_loc = [perm_field.interpolate(s['loc']) for s in params.sources]
    return perms_at_loc

def applying_factors(perm, params):
    """getting multiplication factors for base Q params"""
    # getting back test data
    with h5py.File('model_press_q_testing.h5', 'r') as f:  
        p_k = f["p_k"][:]
        k_list = f["k_list"][:]
        interps = [make_interp_spline(k_list, press, k=1) for press in p_k]

    perms_at_loc = get_perms_at_locs(perm, params)
    for ii, s in enumerate(params.sources):
        k_inplace = perms_at_loc[ii]
        target_diff_P = s['P']
        p_k = interps[ii] # different factors for each well - problem with limit value of P0
        Q_factor = target_diff_P/(p_k(k_inplace) - params.P0) 
        s['Q'] = s['Q'] * Q_factor

    return params