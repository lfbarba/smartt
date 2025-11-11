import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def plot_scalar_slices(scalar_tomogram, poi = None, imshow_args = {}):

    fig, axs = plt.subplots(1,3, figsize = (12, 5))
    
    volume_shape = scalar_tomogram.shape
    if poi is None:
        poi = [volume_shape[0]//2, volume_shape[1]//2, volume_shape[2]//2]
        
    axs[0].imshow(scalar_tomogram[poi[0],:,:],**imshow_args)
    axs[0].set_title('x slice')
    axs[0].set_xlabel('z'); axs[0].set_ylabel('y')
    axs[0].plot([0, volume_shape[2]-0.5], [poi[1]]*2, 'r')
    axs[0].plot([poi[2]]*2, [0, volume_shape[1]-0.5], 'r')
    
    axs[1].imshow(scalar_tomogram[:, poi[1],:], **imshow_args)
    axs[1].set_title('y slice')
    axs[1].set_xlabel('z'); axs[1].set_ylabel('x')
    axs[1].plot([0, volume_shape[2]-0.5], [poi[0]]*2, 'r')
    axs[1].plot([poi[2]]*2, [0, volume_shape[0]-0.5], 'r')
    
    axs[2].imshow(scalar_tomogram[:,:,poi[2]], **imshow_args)
    axs[2].set_title('z slice')
    axs[2].set_xlabel('y'); axs[2].set_ylabel('x')
    axs[2].plot([0, volume_shape[1]-0.5], [poi[0]]*2, 'r')
    axs[2].plot([poi[1]]*2, [0, volume_shape[0]-0.5], 'r')
    
    return fig, axs

def direction_rgb(theta, phi):

    hue = ((phi) % (np.pi))/np.pi
    saturation = (np.arctan(theta/2) / np.arctan(np.pi/4))**2

    modifier = -np.sin(phi)*np.sin(2*theta)**2
    modifier = np.sign(modifier) * np.sqrt(np.abs(modifier))
    value = np.ones(theta.shape)*0.7 + 0.2*modifier
    
    hsv = np.stack([hue, saturation, value], axis = -1)
    return hsv_to_rgb(hsv)


def make_color_wheel(ax, pcolor_opts = {'edgecolors':'face', 'rasterized':True}):

    theta = np.linspace(0, np.pi/2, 100)
    phi = np.linspace(0, 2*np.pi, 400)
    theta, phi = np.meshgrid(theta, phi)

    rgb = direction_rgb(theta, phi)

    img = ax.pcolormesh(phi, np.arctan(theta/2), rgb, **pcolor_opts)
    theta_ticks = np.linspace(0, 90, 4)
    ax.set_yticks(np.arctan(theta_ticks*np.pi/360))
    ax.set_yticklabels([f'{th:.0f}°' for th in theta_ticks])
    return img


def plot_direction_in_slice(main_direction, slice_index, slice_orientation = 'xy', ax = None, mask = None):
    
    volume_shape = main_direction.shape[:3]
    
    if ax is None:
        fig, ax = plt.subplots()
        
    if mask is None:
        mask = np.ones(volume_shape, dtype = bool)
    
    # take care of flipping logic 
    if slice_orientation == 'xy':
        direction_in_slice = main_direction[:, :, slice_index, :]
        mask_slice = mask[:, :, slice_index].transpose()
        where_flip = direction_in_slice[..., 2] < 0.0
        direction_in_slice[where_flip, :] = -direction_in_slice[where_flip, :]
        direction_in_slice = direction_in_slice.transpose((1,0,2))
        theta = np.arccos(direction_in_slice[..., 2])
        phi = np.arctan2(-direction_in_slice[..., 1], direction_in_slice[..., 0])
        extent = (0, volume_shape[0], 0, volume_shape[1])
        
    elif slice_orientation == 'yx':
        direction_in_slice = main_direction[:, :, slice_index, :]
        mask_slice = mask[:, :, slice_index]
        where_flip = direction_in_slice[..., 2] < 0.0
        direction_in_slice[where_flip, :] = -direction_in_slice[where_flip, :]
        theta = np.arccos(direction_in_slice[..., 2])
        phi = np.arctan2(direction_in_slice[..., 0], direction_in_slice[..., 1])
        extent = (0, volume_shape[1], 0, volume_shape[0])
        
    elif slice_orientation == 'xz':
        direction_in_slice = main_direction[:, slice_index, :, :]
        mask_slice = mask[:, slice_index, :].transpose()
        where_flip = direction_in_slice[..., 1] < 0.0
        direction_in_slice[where_flip, :] = -direction_in_slice[where_flip, :]
        direction_in_slice = direction_in_slice.transpose((1,0,2))
        theta = np.arccos(direction_in_slice[..., 1])
        phi = np.arctan2(direction_in_slice[..., 2], direction_in_slice[..., 0])
        extent = (0, volume_shape[0], 0, volume_shape[2])
        
    elif slice_orientation == 'zx':
        direction_in_slice = main_direction[:, slice_index, :, :]
        mask_slice = mask[:, slice_index, :]
        where_flip = direction_in_slice[..., 1] < 0.0
        direction_in_slice[where_flip, :] = -direction_in_slice[where_flip, :]
        theta = np.arccos(direction_in_slice[..., 1])
        phi = np.arctan2(direction_in_slice[..., 0], direction_in_slice[..., 2])   
        extent = (0, volume_shape[2], 0, volume_shape[0])
        
    elif slice_orientation == 'yz':
        direction_in_slice = main_direction[slice_index, :, :, :]
        mask_slice = mask[slice_index, :, :].transpose()
        where_flip = direction_in_slice[..., 0] < 0.0
        direction_in_slice[where_flip, :] = -direction_in_slice[where_flip, :]
        direction_in_slice = direction_in_slice.transpose((1,0,2))
        theta = np.arccos(direction_in_slice[..., 0])
        phi = np.arctan2(direction_in_slice[..., 2], direction_in_slice[..., 1])
        extent = (0, volume_shape[1], 0, volume_shape[2])
        
    elif slice_orientation == 'zy':
        direction_in_slice = main_direction[slice_index, :, :, :]
        mask_slice = mask[slice_index, :, :]
        where_flip = direction_in_slice[..., 0] < 0.0
        direction_in_slice[where_flip, :] = -direction_in_slice[where_flip, :]
        theta = np.arccos(direction_in_slice[..., 0])
        phi = np.arctan2(direction_in_slice[..., 1], direction_in_slice[..., 2])  
        extent = (0, volume_shape[2], 0, volume_shape[1])
        
    rgb = direction_rgb(theta, phi)
    rgb[~mask_slice, :] = 1
    
    ax.imshow(rgb, extent = extent)
    
    return ax, rgb