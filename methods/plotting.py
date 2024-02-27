# plotting func
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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