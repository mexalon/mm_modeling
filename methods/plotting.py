# plotting func
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl

def plotly_plot(model, params=None):
    X, Y, Z = np.indices(np.array(model.shape)) # just idx's
    if params is not None:
        dx, dy, dz = params.dx_dy_dz # meters in one cell
    else:
        dx, dy, dz = 1, 1, 1
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
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
        # norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    # plt.subplots_adjust(wspace=0.4,hspace=0.1)
    # fig.suptitle('Permeability, mD')

    ax1.contourf(x_ax, y_ax, data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm, levels=100)
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ax2.contourf(x_ax, -z_ax, data[:, loc[1], :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax2.set_title('XZ plane')
    # ax2.set_aspect('equal')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.contourf(y_ax, -z_ax, data[loc[0], :, :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax3.set_title('YZ plane')
    # ax3.set_aspect('equal')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
              ax=(ax2, ax3), anchor=(0, 0.5), shrink=1, orientation='vertical', label='Permeability, mD')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()

def plot_press(data, loc, params, vmin_vmax=None, save=False, fname='Pore pressure'):
    cmap = mpl.cm.viridis
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
        # norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    ax1.imshow(data[:, :, loc[2]].transpose(), extent=[x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]], origin='lower', cmap=cmap, norm=norm) 
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ax2.imshow(data[:, loc[1], :].transpose(), extent=[x_ax[0], x_ax[-1], -z_ax[-1], -z_ax[0]], origin='upper', cmap=cmap, norm=norm)
    ax2.set_title('XZ plane')
    # ax2.set_aspect('equal')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.imshow(data[loc[0], :, :].transpose(),  extent=[y_ax[0], y_ax[-1], -z_ax[-1], -z_ax[0]], origin='upper', cmap=cmap, norm=norm)
    ax3.set_title('YZ plane')
    # ax3.set_aspect('equal')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=(ax2, ax3), orientation='vertical', label='Pore pressure, MPa')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()


def plot_events_slice(data, loc, params, vmin_vmax=None, save=False, fname='seism_dens'):
    cmap = mpl.cm.terrain
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
        # norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    ax1.contourf(x_ax, y_ax, data[:, :, loc[2]].transpose(), cmap=cmap, norm=norm, levels=100)
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ax2.contourf(x_ax, -z_ax, data[:, loc[1], :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax2.set_title('XZ plane')
    # ax2.set_aspect('equal')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.contourf(y_ax, -z_ax, data[loc[0], :, :].transpose(),  cmap=cmap, norm=norm, levels=100)
    ax3.set_title('YZ plane')
    # ax3.set_aspect('equal')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=(ax2, ax3), orientation='vertical', label='Seismic density')

    if save:
        plt.savefig(f'{fname}.png', dpi = 300,  bbox_inches='tight', transparent=False)

    plt.show()


def plot_events_projection(data, params, vmin_vmax=None, save=False, fname='seism_dens'):
    cmap = mpl.cm.terrain
    if vmin_vmax is None:
        norm = mpl.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
    else:
        norm = mpl.colors.Normalize(vmin=vmin_vmax[0], vmax=vmin_vmax[1])
        # norm = mpl.colors.LogNorm(vmin=vmin_vmax[0], vmax=vmin_vmax[1])

    x_ax, y_ax, z_ax = (np.linspace(s[0], s[1], sh)/1000 for sh, s in zip(params.shape, params.sides))  # km

    fig = plt.figure(figsize=(7, 3), layout="constrained")
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    (ax2, ax3) = subfigs[1].subplots(2, 1, sharex=ax1)

    ax1.contourf(x_ax, y_ax, np.sum(data, axis=2).transpose(), cmap=cmap, norm=norm, levels=100)
    ax1.set_title('XY plane')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x, km')
    ax1.set_ylabel('y, km')

    ax2.contourf(x_ax, -z_ax, np.sum(data, axis=1).transpose(),  cmap=cmap, norm=norm, levels=100)
    ax2.set_title('XZ plane')
    # ax2.set_aspect('equal')
    ax2.set_xlabel('x, km')
    ax2.set_ylabel('Depth, km')

    ax3.contourf(y_ax, -z_ax, np.sum(data, axis=0).transpose(),  cmap=cmap, norm=norm, levels=100)
    ax3.set_title('YZ plane')
    # ax3.set_aspect('equal')
    ax3.set_xlabel('y, km')
    ax3.set_ylabel('Depth, km')

    # колорбар
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=(ax2, ax3), orientation='vertical', label='Seismic density')

    if save:
        plt.savefig(f'{fname}.png', dpi=300,  bbox_inches='tight', transparent=False)

    plt.show()


def plot_cumulative_events_slice(data, t, loc, params, vmin_vmax=None, save=False, fname='seism_dens'):
    cumsum_data = np.cumsum(data, axis=0)
    plot_events_slice(cumsum_data[t], loc, params, vmin_vmax, save, fname)


def plot_cumulative_events_projection(data, t, params, vmin_vmax=None, save=False, fname='seism_dens'):
    cumsum_data = np.cumsum(data, axis=0)
    plot_events_projection(cumsum_data[t], params, vmin_vmax, save, fname)
    

def plot_event_list(ev_list, params):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    norm = mpl.colors.Normalize(vmin=np.min(ev_list[:,-1]), vmax=np.max(ev_list[:,-1]))
    cmap = mpl.cm.viridis_r
    t, x, y, d, M = [ev_list[:, ii] for ii in range(ev_list.shape[-1])]
    x_m, y_m, d_m = [dxdydz * xyz for dxdydz, xyz in zip(params.dx_dy_dz, [x, y, d])]
    x_km, y_km, d_km = [(bound[0] + xyz)/1000 for bound, xyz in zip(params.sides, [x_m, y_m, d_m])]
    ax.scatter(x_km, y_km, - d_km, marker='o', c=t, cmap=cmap, norm=norm, s=100*M)
    ax.set_aspect('equal')
    ax.set_xlabel('x, km')
    ax.set_ylabel('y, km')
    ax.set_zlabel('Depth, km')
    ax.set_xlim(1e-3*np.array(params.sides[0]))
    ax.set_ylim(1e-3*np.array(params.sides[1]))
    ax.set_zlim(-1e-3*np.array(params.sides[2]))



# ploting tensions
# iii = 0
# fig, ax = plt.subplots()
# ax.scatter(sigma_n.flatten(), tau.flatten(), s=0.1)

# eff_sn = (sigma_n - np.expand_dims(true_pore_press[iii], -1)).flatten()
# ax.scatter(eff_sn.flatten(), tau.flatten(), s=0.1)

# t = seeds.tan_phi_rvs.flatten() * sigma_n.flatten() + seeds.C_rvs.flatten() # C-M crit
# ax.scatter(sigma_n.flatten(), t, s=0.1)
