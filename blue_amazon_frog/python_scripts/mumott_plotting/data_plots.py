import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

def SSAXS_color(angle, doo, intens, max_doo = None, max_intens = None):

    if max_doo is None:
        max_doo = 1
    if max_intens is None:
        max_intens = np.max(intens)

    rgb = hsv_to_rgb(np.stack([
        (angle%np.pi)/np.pi,
        np.clip(0, 1, doo / max_doo)**2,
        np.clip(0, 1, intens / max_intens),], axis = -1)
    )

    return rgb


def make_colorful_plot(data, geometry, ax = None, stepsize = 1, max_doo = None, max_intens = None):
    """ Basic, easy to modify function for making the famous "colorful plot" of scanning SAXS.
    """
    
    if ax is None:
        fig, ax = fig, ax = plt.subplots()


    # Find the angle and a0, a1 numbers.
    a0 = np.sum(data, axis = -1)
    
    if geometry.full_circle_covered == False:
        fourier_component = np.fft.fft(data, axis = -1)[:,:,1]
    elif geometry.full_circle_covered == True:
        fourier_component = np.fft.fft(data, axis = -1)[:,:,2]

    angles = np.angle(fourier_component)
    a1 = np.abs(fourier_component)

    #### This block has to do with converting the angles to the plotting coordinates
    # Angle is currently made under the assumption that the
    # zero-index corresponds to zero-angle. In case there is
    # and offset, we need to correct it.
    sign = np.sign(geometry.detector_angles[1] - geometry.detector_angles[0])
    angles = sign * (angles - geometry.detector_angles[0])

    # Now we need to convert the angle to x-y-z space.
    vectors = np.cos(angles)[:,:,np.newaxis] * geometry.detector_direction_origin[np.newaxis, np.newaxis, :]\
        + np.sin(angles)[:,:,np.newaxis] * geometry.detector_direction_positive_90[np.newaxis, np.newaxis, :]
    # And now we convert back to an angle in plot-space
    angles = np.arctan2(
        np.einsum('i,jki', geometry.j_direction_0, vectors),
        np.einsum('i,jki', geometry.k_direction_0, vectors),
    )

    rgb = SSAXS_color(angles, a1/a0, a0, max_doo, max_intens)
    img = ax.imshow(rgb,
                    extent = (0, geometry.projection_shape[1] * stepsize,
                              0, geometry.projection_shape[0] * stepsize))
    ax.set_ylabel(f'j direction ({geometry.j_direction_0[0]:.0f}, {geometry.j_direction_0[1]:.0f}, {geometry.j_direction_0[2]:.0f})')
    ax.set_xlabel(f'k direction ({geometry.k_direction_0[0]:.0f}, {geometry.k_direction_0[1]:.0f}, {geometry.k_direction_0[2]:.0f})')
    return img, rgb


def make_color_wheel(ax, max_doo = None, pcolor_opts = {'edgecolors':'face', 'rasterized':True}):

    if max_doo is None:
        max_doo = np.max(doo)
    doo = np.linspace(0, max_doo, 100)
    angle = np.linspace(0, 2*np.pi, 400)
    doo, angle = np.meshgrid(doo, angle)

    rgb = SSAXS_color(angle, doo, np.ones(angle.shape), max_doo)

    img = ax.pcolormesh(angle, doo, rgb, **pcolor_opts)
