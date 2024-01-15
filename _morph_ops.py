#!/usr/bin/env python

'''
Widgets for morphological operations on binary images.

'''

import copy
import numpy as np
import cv2
import skimage as ski
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes
from napari.types import LayerDataTuple


@magicgui(
    call_button='Run', 
    operation = {'widget_type': 'ComboBox', 
                 'choices': ['Erode', 'Dilate', 'Open', 'Close', 
                            'WhiteTopHat', 'BlackTopHat']},
    kernel_type = {'widget_type': 'ComboBox', 
                   'choices': ['Rectangle', 'Cross', 'Ellipse']},
    kernel_width = {'widget_type': 'SpinBox', 'step': 1},
    kernel_height = {'widget_type': 'SpinBox', 'step': 1},
    name={'label': 'Output image layer name'}
    )
def morph_op(image: Image, 
        operation: str,
        kernel_type: str='Rectangle',
        kernel_width: int=3, 
        kernel_height: int=0, 
        name: str='Morphed'   
       ) -> LayerDataTuple :

    """
    Performs a morphology operation. Wraps ``cv::morphologyEx()`` from OpenCV.
    See https://docs.opencv.org/4.8.0/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f

    Parameters
    ----------
    image : napari.layers.Image
        Input binary image (datatype bool)
    operation : str 
        Type of operation: {'Erode', 'Dilate', 'Open', 'Close', 'WhiteTopHat',
        'BlackTopHat'}
    kernel_type : str
        Type of the kernel: {'Rectangle', 'Cross', 'Ellipse'}
    kernel_width : int
        If zero, `kernel_height` must be > 0 and `kernel_width` will be set
        equal to `kernel_height`. Both `kernel_width` and `kernel_height` cannot
        be zero.
    kernel_height : int
        If zero, `kernel_width` must be > 0 and `kernel_height` will be set
        equal to `kernel_width`. Both `kernel_width` and `kernel_height` cannot
        be zero.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Morphed image with bool datatype

    """
    morph_ops = {'Erode': cv2.MORPH_ERODE, 
                 'Dilate': cv2.MORPH_DILATE,
                 'Open': cv2.MORPH_OPEN, 
                 'Close': cv2.MORPH_CLOSE, 
                 'WhiteTopHat': cv2.MORPH_TOPHAT,
                 'BlackTopHat': cv2.MORPH_BLACKHAT}

    morph_shapes = {'Rectangle': cv2.MORPH_RECT, 'Cross': cv2.MORPH_CROSS,
                    'Ellipse': cv2.MORPH_ELLIPSE}

    assert ( (kernel_width != 0) or (kernel_height != 0) )
    if kernel_width == 0:
        kernel_width = kernel_height
    if kernel_height == 0:
        kernel_height = kernel_width

    kernel = cv2.getStructuringElement(morph_shapes[kernel_type], 
                                       (kernel_width, kernel_height))
    img = ski.util.img_as_ubyte(image.data)
    img_dst = cv2.morphologyEx(img, morph_ops[operation], kernel)
    img_dst = (img_dst > 0)
    return (img_dst, 
            {'name': name, 'metadata': image.metadata,
             'colormap': 'gray', 'interpolation2d': 'nearest'},
            'image')




@magicgui(call_button='Run',
          name={'label': 'Output image layer name'})
def remove_objects(image: Image,
                   min_size: int=64,
                   labels: Labels=None,
                   shapes: Shapes=None,
                   name: str='ObjectsRemoved'
                   ) -> LayerDataTuple:
    """
    Removes small foreground objects from a binary image. The foreground is
    assumed to be white (i.e. True pixels) and the background is assumed to be
    black (i.e. False pixels).
    Wraps ``remove_small_objects()`` from scikit-image. 
    See https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_objects

    Parameters
    ----------
    image : napari.layers.Image
        Input binary image with datatype bool
    min_size : int
        Objects smaller than `min_size` will be removed.
    labels : napari.layers.Labels, optional
        Napari labels layer. If present, all pixels labeled as `1` will be
        marked False.  
    shapes : napari.layers.Shapes, optional
        Napari shapes layer. If present, all pixels contained in any shape will
        be marked False.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Image with bool datatype after removing objects.

    """
    assert np.issubdtype(image.data.dtype, bool)

    if (labels is None) and (shapes is None):
        img = ski.morphology.remove_small_objects(image.data, min_size,
                                                connectivity=1)
    else:
        img = copy.deepcopy(image.data)
        if labels:
            img[labels.data == 1 ] = False
        if shapes:
            for each in shapes.data:
                rr, cc = ski.draw.polygon(each[:,0], each[:,1],
                                                shape=image.data.shape)
                img[rr,cc] = False
    return (img, {'name': name, 'metadata': image.metadata,
                  'colormap': 'gray', 'interpolation2d': 'nearest'}, 'image')



@magicgui(call_button='Run',
          name={'label': 'Output image layer name'})
def fill_holes(image: Image,
                max_area: int=64,
                labels: Labels=None,
                shapes: Shapes=None,
                name: str='HolesFilled'
                ) -> LayerDataTuple:
    """
    Fills small background holes from foreground objects in a binary image.
    The foreground is assumed to be white (i.e. True pixels) and the background
    is assumed to be black (i.e. False pixels).
    Wraps ``remove_small_holes()`` from scikit-image. 

    See https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_holes

    Parameters
    ----------
    image : napari.layers.Image
        Input binary image with datatype bool
    max_area : int
        Holes smaller than `max_area` will be removed.
    labels : napari.layers.Labels, optional
        Napari labels layer. If present, all pixels labeled as `1` will be
        marked True.  
    shapes : napari.layers.Shapes, optional
        Napari shapes layer. If present, all pixels contained in any shape will
        be marked True.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Image with bool datatype after filling holes.

    """
    assert np.issubdtype(image.data.dtype, bool)

    if (labels is None) and (shapes is None):
        img = ski.morphology.remove_small_holes(image.data, 
                                        max_area, connectivity=1)
    else:
        img = copy.deepcopy(image.data)
        if labels:
            img[labels.data == 1 ] = True
        if shapes:
            for each in shapes.data:
                rr, cc = ski.draw.polygon(each[:,0], each[:,1],
                                            shape=image.data.shape)
                img[rr,cc] = True
    return (img, {'name': name, 'metadata': image.metadata, 'colormap': 'gray',
                  'interpolation2d': 'nearest'}, 'image')


morph_op_widgets = [(morph_op, 'Morphology Operation'),
                    (remove_objects, 'Remove Objects'),
                    (fill_holes, 'Fill Holes')
                   ]
