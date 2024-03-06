# plotting func
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl

def plotly_plot(model):
    X, Y, Z = np.indices(np.array(model.shape)) # just idx's
    dx, dy, dz = 200, 200, 200 # meters in one cell
    X = X*dx
    Y = Y*dy
    Z = - Z*dz # depth

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=model.flatten(),

        opacity=0.3, 
        surface_count=21, # needs to be a large number for good volume rendering
        ))

    fig.show()


def plot_perm(data, loc, params, vmin_vmax=None, save=False, fname='permeability'):
    cmap = mpl.cm.Set2_r
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(0, sl, sh)/1000 for sh, sl in zip(params.shape, params.side_lenght))  # km

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
    plt.subplots_adjust(wspace=0.4,hspace=0.1)
    # fig.suptitle('Permeability, mD')

    ax1.contourf(x_ax, y_ax, data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm, levels=100)
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ax2.contourf(x_ax, -z_ax, data[:, loc[1], :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax2.set_title('XZ plane')
    ax2.set_aspect('equal')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.contourf(y_ax, -z_ax, data[loc[0], :, :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax3.set_title('YZ plane')
    ax3.set_aspect('equal')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
              ax=(ax1, ax2, ax3), anchor=(-0.2, 0.5), shrink=0.85, orientation='vertical', label='Permeability, mD')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()

def plot_press(data, loc, params, vmin_vmax=None, save=False, fname='Pore pressure'):
    cmap = mpl.cm.viridis
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(0, sl, sh)/1000 for sh, sl in zip(params.shape, params.side_lenght))  # km

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
    plt.subplots_adjust(wspace=0.4,hspace=0.1)
    # fig.suptitle('Pressure, mPa')

    ax1.imshow(data[:, :, loc[2]].transpose(), extent=[x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]], origin='lower', cmap=cmap, norm=norm) 
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ax2.imshow(data[:, loc[1], :].transpose(), extent=[x_ax[0], x_ax[-1], -z_ax[-1], z_ax[0]], origin='upper', cmap=cmap, norm=norm)
    ax2.set_title('XZ plane')
    ax2.set_aspect('equal')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.imshow(data[loc[0], :, :].transpose(),  extent=[y_ax[0], y_ax[-1], -z_ax[-1], z_ax[0]], origin='upper', cmap=cmap, norm=norm)
    ax3.set_title('YZ plane')
    ax3.set_aspect('equal')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
              ax=(ax1, ax2, ax3), anchor=(-0.2, 0.5), shrink=0.85, orientation='vertical', label='Pore pressure, MPa')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()