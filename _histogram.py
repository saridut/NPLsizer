#!/usr/bin/env python

'''
Widgets for histogram manipulation of grayscale images.

'''

import skimage as ski
from typing import Tuple
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes
from napari.types import LayerDataTuple


@magicgui(call_button='Run',
          name={'label': 'Output image layer name'}
         )
def adjust_gamma(image: Image,
        gamma: float=1.0,
        gain: float=1.0,
        name: str='GammaCorrected') -> LayerDataTuple: 
    """
    Performs Gamma correction on a grayscale image. Wraps ``adjust_gamma()`` from
    scikit-image. 

    See https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma

    Parameters
    ----------
    image : napari.layers.Image
        Input image
    gamma : float
        Non negative real number. For `gamma > 1`,  the output image will be
        darker than the input image. For `gamma < 1`, he output image will be
        brighter than the input image. 
    gain : float
        The constant multiplier.
    name : str
        Name of output napari image layer 

    Returns
    -------
    napari.types.LayerDataTuple
        Gamma corrected image with same datatype as input.

    """
    img = ski.exposure.adjust_gamma(image.data, gamma=gamma, gain=gain)
    return (img, {'name': name, 'metadata': image.metadata, 'colormap': 'gray',
                     'interpolation2d': 'spline36'}, 'image')



@magicgui(call_button='Run', name={'label': 'Output image layer name'})
def equalize_hist(image: Image, 
        nbins: int=256,
        name: str='HistEqualized') -> LayerDataTuple :
    """
    Performs histogram equalization for a grayscale image. Wraps
    ``equalize_hist()`` from scikit-image. 

    See https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    nbins : int
        Number of bins
    name : str
        Name of output napari image layer 

    Returns
    -------
    napari.types.LayerDataTuple
        Histogram equalized image with 64-bit float datatype.

    """
    img = ski.exposure.equalize_hist(image.data, nbins=nbins)
    return (img, {'name': name, 'metadata': image.metadata, 'colormap': 'gray', 
                     'interpolation2d': 'spline36'}, 'image')



@magicgui(call_button='Run',
          nreg_width={'widget_type': 'SpinBox', 'min': 1, 'step': 1},
          nreg_height={'widget_type': 'SpinBox', 'step': 1},
          clip_limit={'widget_type': 'FloatSlider', 'min': 0, 'max': 1.0, 
                      'step': 0.01},
          name={'label': 'Output image layer name'}
          )
def equalize_histadapt(image: Image, 
        nreg_width: int=8, 
        nreg_height: int=0, 
        clip_limit: float=0.05, 
        nbins: int=256,
        name: str='AdaptHistEqualized') -> LayerDataTuple :

    """
    Performs CLAHE for a grayscale image. Wraps ``equalize_adapthist()`` from
    scikit-image. 

    See https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    nreg_width : int
        Number of regions along the width, must be > 1
    nreg_height : int
        Number of regions along the height. If <= 0, will be set equal to
        `nreg_width`.
    clip_limit : float
        Clipping limit, normalized between 0 and 1 (higher values give more contrast).
    nbins : int
        Number of gray bins for histogram
    name : str
        Name of output napari image layer 

    Returns
    -------
    napari.types.LayerDataTuple
        Histogram equalized image with 64-bit float datatype.

    """
    if nreg_height <= 0:
        nreg_height = nreg_width

    m, n = image.data.shape
    ks = (m//nreg_height, n//nreg_width)
    img = ski.exposure.equalize_adapthist(image.data, kernel_size=ks, 
                                          clip_limit=clip_limit, nbins=nbins)
    return (img, {'name': name, 'metadata': image.metadata, 'colormap': 'gray',
                     'interpolation2d': 'spline36'}, 'image')



hist_widgets = [(adjust_gamma, 'Gamma correction'),
                (equalize_hist, 'Equalize Histogram'),
                (equalize_histadapt, 'CLAHE')
                ]

