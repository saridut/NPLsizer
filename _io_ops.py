#!/usr/bin/env python

'''
Widgets for reading in grayscale images and writing post-segmentation
results.

'''

import os
import copy
import math
import pathlib
import numpy as np
import skimage as ski
from magicgui import magicgui
from napari.types import LayerDataTuple
from napari.layers import Image, Shapes
import napari
import dm3_lib as dm3


#Pixel size for dm3 files, defaults to 1.0 for all other file types.
PXSIZE = 1.0
#Pixel unit for dm3 files, defaults to 'px' for all other file types.
PXUNIT = 'px'


@magicgui(call_button='Open',
          fn = {'label': 'Image file', 'mode': 'r'},
          write_dm3_metadata = {'label': 'Write metadata for DM3 files'},
          fn_dm3_metadata = {'label': 'DM3 metadata file', 'mode': 'w'},
          name={'label': 'Image layer name'}
          )
def read_image(fn: pathlib.Path,
               write_dm3_metadata: bool=False,
               fn_dm3_metadata: pathlib.Path='metadata.txt',
               name: str=''
               ) -> LayerDataTuple:

    """ Reads in an image file. 

    The image file may by a numpy `.npy` file.

    `Dm3` files are read using the `dm3_lib`
    package. All other file types are passed on to the `imread` function of
    scikit-image.

    The input image is expected to be grayscale, which will be rescaled to
    64-bit floats in [0,1]. Boolean arrays will be read as is. RGB images will
    be converted to grayscale (64-bit floats in [0,1]).

    Parameters
    ----------
    fn : str or pathlib.Path
        Image file name, e.g. ``test.dm3``.
    write_dm3_metadata : bool
        In case of ``.dm3`` files whether to write metadata to a separate file.
    fn_dm3_metadata : str or pathlib.Path
        Metadata file for ``.dm3`` files. 
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        A napari image layer

    """
    root, ext = os.path.splitext(fn)
    ft = ext.lower().lstrip('.')

    basename = os.path.basename(root)
    if not name:
        name = basename

    global PXSIZE, PXUNIT
    if ft == 'npy':
        img = np.load(fn)
        if np.issubdtype(img.dtype, np.floating):
            img = ski.exposure.rescale_intensity(img)
        if np.issubdtype(img.dtype, np.integer):
            img = ski.exposure.rescale_intensity(img, out_range=np.float64)
        PXSIZE = 1.0; PXUNIT = 'px'
    elif ft == 'dm3':
        img_dm3 = dm3.DM3(fn)
        img = ski.exposure.rescale_intensity(img_dm3.imagedata, out_range=np.float64)
        PXSIZE = img_dm3.pxsize[0]
        PXUNIT = img_dm3.pxsize[1].decode()
        if write_dm3_metadata:
            mdata = {'filename': img_dm3.filename,
                     'width': img_dm3.width, 
                     'height': img_dm3.height, 
                     'depth': img_dm3.depth,
                     'datatype': img_dm3.data_type_str,
                     'pixel_size': PXSIZE,
                     'pixel_unit': PXUNIT
                     }
            for key, val in img_dm3.info.items():
                vald = val.decode(); mdata[key] = vald
            with open(fn_dm3_metadata, 'w') as fh:
                for key, val in mdata.items():
                    fh.write('%s : %s\n'%(key, val))
    else :
        img = ski.io.imread(fn, as_gray=True)
        PXSIZE = 1.0; PXUNIT = 'px'
    assert img.ndim == 2 
    return (img, {'name': name, 'colormap': 'gray', 
                  'interpolation2d': 'spline36'}, 'image')



@magicgui(call_button='Save',
          fn = {'label': 'Output file', 'mode': 'w'},
          jpeg_quality = {'label': 'JPEG compression', 'widget_type': 'Slider',
                          'min': 1, 'max': 100, 'step': 1}
          )
def write_image(image: Image,
               fn: pathlib.Path,
               jpeg_quality: int=75
               ) -> None:

    """Writes an image layer to a file. The file type depends on the filename 
    extension. Use ``.npy`` to save as a numpy array.

    Calls `imsave` from scikit-image. 

    See https://scikit-image.org/docs/0.20.x/api/skimage.io.html#skimage.io.imsave

    Parameters
    ----------
    image : napari.layers.Image
        Name of the napari image layer
    fn : str or pathlib.Path
        Output file name
    jpeg_quality : int
        For writing to a jpeg file, compression quality. Can be any value from 1
        to 100, where 1 is the worst and 100 is the best.

    Returns
    -------
    None

    """
    root, ext = os.path.splitext(fn)
    ft = ext.lower().lstrip('.')

    if ft == 'npy':
        np.save(fn, image.data)
    else :
        print(image.data.dtype)
        if np.issubdtype(image.data.dtype, np.bool_):
            img = ski.util.img_as_ubyte(image.data)
        else:
            img = image.data
        ski.io.imsave(fn, img, check_contrast=False, quality=jpeg_quality)



@magicgui(call_button='Run', 
          fn_out={'label': 'Output file', 'mode':'w'})
def calc_size(shapes: Shapes,
              fn_out: pathlib.Path=None,
              clear_shapes: bool=True
              ) -> None :
    """
    Given a napari shapes layer containing a set of rectangles, calculates
    their lengths, heights, and areas.

    Parameters
    ----------
    shapes : napari.layers.Shapes
        Napari shapes layer containing zero or more rectangles.
    fn_out : str or pathlib.Path
        Output file name with extension ``.csv``.
    clear_shapes : bool
        Whether to clear all shapes in the shape layer

    Returns
    -------
    None

    """
    #Calculate the length of the two sides of each rectangle
    #for each in shapes.data:
    nshapes = shapes.nshapes
    rect_sizes = np.zeros((nshapes,2))
    for i in range(nshapes):
        if shapes.shape_type[i] != 'rectangle':
            continue
        verts = shapes.data[i]
        d = [np.linalg.norm(verts[1,:] - verts[0,:]),
             np.linalg.norm(verts[2,:] - verts[0,:]),
             np.linalg.norm(verts[3,:] - verts[0,:])]
        d.sort()
        rect_sizes[i,0] = d[1]; rect_sizes[i,1] = d[0]

    rect_sizes *= PXSIZE
    with open(fn_out, 'w') as fh:
        fh.write(f'Unit, {PXUNIT}, \n')
        fh.write(f'length, height, area\n')
        for i in range(nshapes):
            fh.write( '%g, %g, %g\n'%(rect_sizes[i,0], rect_sizes[i,1],
                rect_sizes[i,0]*rect_sizes[i,1]) )
    if clear_shapes:
        shapes.data = []



io_op_widgets = [
                 (read_image, 'Read image'),
                 (write_image, 'Write image'),
                 (calc_size, 'Get particle size')
                ]
