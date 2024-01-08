#!/usr/bin/env python

'''
Widgets for thresholding grayscale images.

'''
import numpy as np
import cv2
import skimage as ski
from magicgui import magicgui
from napari.layers import Image
from napari.types import LayerDataTuple


@magicgui(call_button='Run', 
    name={'label': 'Output image layer name'}
    )
def threshold(image: Image,
              low_is_white: bool=True,
              name: str='Binarized'
              ) -> LayerDataTuple:
    """
    Performs thresholding using Otsu's method.
    Wraps ``cv::threshold()`` from OpenCV. 

    See https://docs.opencv.org/4.8.0/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    low_is_white : bool
        Should low intensity pixels be white ?
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Binarized image with np.bool datatype

    """
    thresh_type = cv2.THRESH_BINARY_INV if low_is_white else cv2.THRESH_BINARY 
    thresh_type +=  cv2.THRESH_OTSU
    img = ski.util.img_as_ubyte(image.data)
    thres_val, img_threshed = cv2.threshold(img, 0, 255, thresh_type)
    img_bin = (img_threshed > 0)
    return (img_bin, 
            {'name': name, 'colormap': 'gray', 'interpolation2d': 'nearest'},
            'image')



@magicgui(call_button='Run', 
    adaptive_method={'widget_type': 'ComboBox', 'choices': ['Gaussian', 'Mean']},
    block_size={'widget_type': 'SpinBox', 'min': 3, 'step': 2},
    const={'widget_type': 'FloatSpinBox', 'min': -100.0},
    name={'label': 'Output image layer name'}
    )
def adaptive_threshold(image: Image,
            low_is_white: bool=True,
            adaptive_method: str='Gaussian',
            block_size: int=11,
            const: float=2.0,
            name: str='BinarizedAdapThresh'
            ) -> LayerDataTuple:
    """
    Performs thresholding using an adaptive method.
    Wraps ``cv::adaptiveThreshold()`` from OpenCV. 

    See https://docs.opencv.org/4.8.0/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    low_is_white : bool
        Should low intensity pixels be white ?
    adaptive_method : {'Gaussian', 'Mean'}
        If the threshold value is a mean or gaussian-weighted sum over
        neighbouring pixels. 
    block_size : int
        Size of a pixel neighborhood that is used to calculate a threshold value
        for the pixel; must be odd.
    const : float
        Constant subtracted from the mean or weighted mean. Normally, it is
        positive but may be zero or negative as well.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Binarized image with np.bool datatype

    """
    if adaptive_method == 'Gaussian':
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif adaptive_method == 'Mean':
        method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        raise ValueError('Bad value for `method`: %s'%method)

    thresh_type = cv2.THRESH_BINARY_INV if low_is_white else cv2.THRESH_BINARY 

    img = ski.util.img_as_ubyte(image.data)
    img_threshed = cv2.adaptiveThreshold(img, 255, method, thresh_type,
                                        block_size, const)
    img_bin = (img_threshed > 0)
    return (img_bin, 
            {'name': name, 'colormap': 'gray', 'interpolation2d': 'nearest'},
            'image')



threshold_widgets = [(threshold, 'Threshold'),
                     (adaptive_threshold, 'Adaptive Threshold')
                    ]
