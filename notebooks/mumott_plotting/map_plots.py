import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_on_polefigure(ax, intensity, theta, phi, pcolor_opts = {'edgecolors':'face', 'rasterized':True, 'cmap':'jet'}):
    img = ax.pcolormesh(phi, np.arctan(theta/2), intensity, **pcolor_opts)
    theta_ticks = np.linspace(0, 90, 4)
    ax.set_yticks(np.arctan(theta_ticks*np.pi/360))
    ax.set_yticklabels([f'{th:.0f}°' for th in theta_ticks])
    
    return img



def make_scattering_vector_plot(ax, intensity, theta, phi, pcolor_opts = {'edgecolors':'face', 'rasterized':False}):

    # Cast angles to vectors
    X = intensity*np.sin(theta) * np.cos(phi)
    Y = intensity*np.sin(theta) * np.sin(phi)
    Z = intensity*np.cos(theta)
    
    norm = plt.Normalize(vmin=intensity.min(), vmax=intensity.max()) 
    surface=ax.plot_surface(X, Y, Z, cstride=1, rstride=1, 
                            facecolors=cm.jet(norm(intensity)))

    ax.axis('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    
    return surface