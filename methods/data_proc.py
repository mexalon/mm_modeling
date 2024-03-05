# func to data proc
import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator


def downscale(inp, factor: tuple):
    '''
    The running average is calculated for all points in the input array, 
    so that the final array contains the mean value. 
    Subsequently, interpolation is performed onto a new grid, with the step specified by a tuple called the factor: (2,2,2) for downscaling by 2 times and so on.
    '''
    inp_dtype = inp.dtype
    avg_inp = ndimage.uniform_filter(inp.astype('float32'), size=factor, mode='nearest') #  it works with float32 only

    x, y, z = (np.arange(0,k,1) for k in avg_inp.shape) # old mesh
    interp = RegularGridInterpolator((x,y,z), avg_inp) # interpolator object

    kx, ky, kz = (np.arange(0,k,f) for k, f in zip(avg_inp.shape, factor)) 
    m1, m2, m3 = np.meshgrid(kx, ky, kz, indexing='ij') # new mesh
    out = interp((m1, m2, m3)) # downscaled array
    return out.astype(inp_dtype)

def downscale_to_shape(inp, new_shape: tuple):
    '''
    The running average is calculated for all points in the input array, 
    so that the final array contains the mean value. 
    Subsequently, interpolation is performed onto a new grid with "new_shape" shape.
    '''
    inp_dtype = inp.dtype
    factor = (i//s for i, s in zip(inp.shape, new_shape))
    avg_inp = ndimage.uniform_filter(inp.astype('float32'), size=factor, mode='nearest') #  it works with float32 only

    x, y, z = (np.arange(0,k,1) for k in avg_inp.shape) # old mesh
    interp = RegularGridInterpolator((x,y,z), avg_inp) # interpolator object

    kx, ky, kz = (np.linspace(0,k-1,f) for k, f in zip(avg_inp.shape, new_shape)) 
    m1, m2, m3 = np.meshgrid(kx, ky, kz, indexing='ij') # new mesh

    out = interp((m1, m2, m3)) # downscaled array
    return out.astype(inp_dtype)

