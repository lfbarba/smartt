import numpy as np

def GK_mapper(basis_set, coefficients, theta_pts, phi_pts):
    """
    Function to calculate basis function maps of the GaussianKernel basis set.
    Parameters:
        - basis_set: the Gaussian kernel basis set.
        - theta_pts, phi_pts: number of theta and phi points on the full sphere of the plotting grid.
    Returns:
        - map_GK: function on defined grid [theta_pts + 1, phi_pts + 1]
        - theta_plot: theta mesh (whole sphere) [theta_pts + 1, phi_pts + 1]
        - phi_plot: phi mesh (whole sphere) [theta_pts + 1, phi_pts + 1]
        - plot_grid_vectors: plotting grid [theta_pts + 1, phi_pts + 1, 3]
    """
    # calculate xyz of basis set vectors 
    theta, phi = basis_set.grid # theta: [0,pi/2], phi: [0,2*pi]
    
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi) 
    Z = np.cos(theta) 

    basis_set_vectors = np.stack((X, Y, Z)) #distributed on half sphere

    #calculate xyz of plot mesh
    res = 1
    polar_coordinates = np.linspace(0, np.pi, theta_pts//res+1)
    azimuthal_coordinates = np.linspace(-np.pi, np.pi, phi_pts//res+1) # I add one extra point at 360degrees to avoid a gap.
    theta_plot, phi_plot = np.meshgrid(polar_coordinates,
                             azimuthal_coordinates,
                             indexing='ij')
    plot_shp = theta_plot.shape

    X_plot = np.sin(theta_plot.flatten()) * np.cos(phi_plot.flatten())
    Y_plot = np.sin(theta_plot.flatten()) * np.sin(phi_plot.flatten()) 
    Z_plot = np.cos(theta_plot.flatten()) 
    vectors_plot = np.stack((X_plot, Y_plot, Z_plot))

    X = np.sin(theta_plot) * np.cos(phi_plot)
    Y = np.sin(theta_plot) * np.sin(phi_plot)
    Z = np.cos(theta_plot) 

    plot_grid_vectors = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=2) #have x,y,z values in index 2

    # calculate distance matrix between both sets
    dotprod = np.einsum("ji,jk->ik",basis_set_vectors, vectors_plot)
    dotprod[np.abs(dotprod) > 1] = 1 #for numerical errors leading to dot products > 1
    distances = np.arccos(np.abs(dotprod)).T
    np.fill_diagonal(distances, 0) 

    # calculate distance matrix of basis set vectors
    dotprod = np.einsum("ji,jk->ik",basis_set_vectors, basis_set_vectors)
    dotprod[np.abs(dotprod) > 1] = 1 #for numerical errors leading to dot products > 1
    mesh_distances = np.arccos(np.abs(dotprod)).T
    np.fill_diagonal(mesh_distances, 0)

    # calculate the basis function maps of the gaussian kernels
    std = (basis_set._kernel_scale_parameter * np.sqrt(2 * np.pi)) / (2 * (basis_set._grid_scale + 1))
    matrix = np.exp(-(1 / 2) * (distances / std) ** 2)
    norm_matrix = np.exp(-(1 / 2) * (mesh_distances / std) ** 2)
    # The normalization factor is the inverse of the unnormalized function value at each grid point
    norm_factors = np.reciprocal(norm_matrix.sum(-1).reshape(1, 1, -1))
    norm_factors = np.ones_like(norm_factors)

    basis_function_maps =  matrix * norm_factors
    basis_function_maps=basis_function_maps[0, :, :]

    map_GK = np.sum(basis_function_maps * coefficients[np.newaxis, :], axis = -1)
    map_GK = map_GK.reshape(plot_shp)

    return map_GK, theta_plot, phi_plot, plot_grid_vectors

def RF_mapper(basis_set, coefficients, theta_pts, phi_pts):
    """
    Function to calculate basis function maps of the GaussianKernel basis set.
    Parameters:
        - basis_set: the Gaussian kernel basis set.
        - theta_pts, phi_pts: number of theta and phi points on the full sphere of the plotting grid.
    Returns:
        - map_GK: function on defined grid [theta_pts + 1, phi_pts + 1]
        - theta_plot: theta mesh (whole sphere) [theta_pts + 1, phi_pts + 1]
        - phi_plot: phi mesh (whole sphere) [theta_pts + 1, phi_pts + 1]
        - plot_grid_vectors: plotting grid [theta_pts + 1, phi_pts + 1, 3]
    """
    # calculate xyz of basis set vectors 
    theta, phi = basis_set.grid # theta: [0,pi/2], phi: [0,2*pi]
    
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi) 
    Z = np.cos(theta) 

    basis_set_vectors = np.stack((X, Y, Z)) #distributed on half sphere

    #calculate xyz of plot mesh
    res = 1
    polar_coordinates = np.linspace(0, np.pi, theta_pts//res+1)
    azimuthal_coordinates = np.linspace(-np.pi, np.pi, phi_pts//res+1) # I add one extra point at 360degrees to avoid a gap.
    theta_plot, phi_plot = np.meshgrid(polar_coordinates,
                             azimuthal_coordinates,
                             indexing='ij')
    plot_shp = theta_plot.shape

    X_plot = np.sin(theta_plot.flatten()) * np.cos(phi_plot.flatten())
    Y_plot = np.sin(theta_plot.flatten()) * np.sin(phi_plot.flatten()) 
    Z_plot = np.cos(theta_plot.flatten()) 
    vectors_plot = np.stack((X_plot, Y_plot, Z_plot))

    X = np.sin(theta_plot) * np.cos(phi_plot)
    Y = np.sin(theta_plot) * np.sin(phi_plot)
    Z = np.cos(theta_plot) 

    plot_grid_vectors = np.concatenate((X[..., None], Y[..., None], Z[..., None]), axis=2) #have x,y,z values in index 2

    # calculate distance matrix between both sets
    dotprod = np.einsum("ji,jk->ik",basis_set_vectors, vectors_plot)
    dotprod[np.abs(dotprod) > 1] = 1 #for numerical errors leading to dot products > 1
    distances = np.arccos(np.abs(dotprod)).T
    np.fill_diagonal(distances, 0) 

    # calculate distance matrix of basis set vectors
    dotprod = np.einsum("ji,jk->ik",basis_set_vectors, basis_set_vectors)
    dotprod[np.abs(dotprod) > 1] = 1 #for numerical errors leading to dot products > 1
    mesh_distances = np.arccos(np.abs(dotprod)).T
    np.fill_diagonal(mesh_distances, 0)

    matrix = np.exp(-(1 / 2) * ((np.pi/2 - distances) / basis_set._ring_width_parameter) ** 2)
    
    basis_function_maps = matrix

    map_GK = np.sum(basis_function_maps * coefficients[np.newaxis, :], axis = -1)
    map_GK = map_GK.reshape(plot_shp)

    return map_GK, theta_plot, phi_plot, plot_grid_vectors

def find_pts(phi_pick, theta_pick, phi_plot, theta_plot, distance):
    """
    Function to find GaussianKernel plot grid vectors within a defined distance of a specific direction.
    Parameters:
        - phi_pick, theta_pick: phi and theta angle of the chosen direction
        - phi_plot, theta_plot: phi and theta meshes output from GK_mapper (whole sphere)
        - distance: maximum distance of returned pts to chosen direction
    Returns:
        - pts: list of arguments of phi_plot and theta_plot of points within distance of specific direction
    """
    # calculate x,y,z of specific direction
    X = np.sin(theta_pick) * np.cos(phi_pick)
    Y = np.sin(theta_pick) * np.sin(phi_pick) 
    Z = np.cos(theta_pick) 
    vector = np.stack((X, Y, Z))
    vector = vector[:,np.newaxis]

    # calculate x,y,z of plot grid (whole sphere)
    plot_shp = theta_plot.shape
    X_plot = np.sin(theta_plot.flatten()) * np.cos(phi_plot.flatten())
    Y_plot = np.sin(theta_plot.flatten()) * np.sin(phi_plot.flatten()) 
    Z_plot = np.cos(theta_plot.flatten()) 
    vectors_plot = np.stack((X_plot, Y_plot, Z_plot))

    # calculate distance of each plot grid vector to the chosen direction
    dotprod = np.einsum("ji,jk->ik", vector, vectors_plot)
    distances = np.arccos(np.abs(dotprod)).T

    distances = distances.reshape(plot_shp)
    # find points on halfsphere within the distance
    pts = np.argwhere(distances[:90,:] < distance)

    return pts
