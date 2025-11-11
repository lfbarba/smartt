import numpy as np
import matplotlib.pyplot as plt


def make_scattering_vector_plot(geometry, projection_index, ax = None):
    
    if ax is None:
        ax = plt.subplot(1,1,1, projection='3d')
    
    probed_coordinates = geometry._get_probed_coordinates().vector
    print(probed_coordinates.shape)
    probed_theta_interval = np.arccos(probed_coordinates[..., 2])
    probed_phi_interval = np.arctan2(probed_coordinates[..., 1], probed_coordinates[..., 0])
    
    # PLot shadows
    for det_seg in range(len(geometry.detector_angles)):
        t = probed_theta_interval[projection_index,det_seg,:]
        p = probed_phi_interval[projection_index,det_seg,:]
        ## WARNING PLOTTING IN X-Z-Y
        ax.plot(np.sin(t)*np.cos(p),np.cos(t),[-1]*len(t), 'k.')
        ax.plot(np.sin(t)*np.cos(p),[1]*len(t),np.sin(t)*np.sin(p), 'k.')
        ax.plot([-1]*len(t),np.cos(t),np.sin(t)*np.sin(p), 'k.')
    
    for det_seg in range(len(geometry.detector_angles)):
        
        t = probed_theta_interval[projection_index,det_seg,:]
        p = probed_phi_interval[projection_index,det_seg,:]
        
        ## WARNING PLOTTING IN X-Z-Y, because I like the defaul viewing angle better there
        ax.plot(np.sin(t)*np.cos(p),np.cos(t),np.sin(t)*np.sin(p),linewidth = 4)
        

    
    p_direction = geometry[projection_index].rotation.T @ geometry.p_direction_0
    ax.quiver(-p_direction[0], -p_direction[2], -p_direction[1], 2*p_direction[0], 2*p_direction[2], 2*p_direction[1],color='k')

    ax.axis('equal')
    ax.set_zlim((-1, 1)); ax.set_ylim((-1, 1)); ax.set_xlim((-1, 1))
    ax.set_zticks((-1, -0.5, 0, 0.5, 1)); ax.set_yticks((-1, -0.5, 0, 0.5, 1)); ax.set_xticks((-1, -0.5, 0, 0.5, 1))
    ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('y')
    ax.set_title(f'Probed part of reciprocal space for\nProjection #{projection_index} in sample-space')
    
    
    return ax
