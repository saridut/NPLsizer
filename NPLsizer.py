#!/usr/bin/env python

'''
Driver script for calculating nanoplatelet size from an image.

'''

import os
import copy
import math
import pathlib
import numpy as np
import cv2
import skimage as ski
from typing import Tuple, Union, List
from magicgui import magicgui
import napari
import dm3_lib as dm3 # https://github.com/piraynal/pyDM3reader

PXSIZE = 1.0
PXUNIT = 'nm'
FN_ROOT = None


def run_watershed(img, tag_fg=2, tag_marker=3, ks=3):
    #img : binary image (uint8) with markers

    img_ws = np.zeros(img.shape, dtype=np.uint8)
    img_ws[img==tag_fg] = 255
    img_ws[img==tag_marker] = 255

    #Dilate the image to create sure background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks,ks))
    sure_bg = cv2.dilate(img_ws, kernel, iterations=1)

    #Create sure foregound
    sure_fg = np.zeros(img.shape, dtype=np.uint8)
    sure_fg[img == tag_marker] = 255

    #The unknown region to segment with watershed
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label the markers
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is 1
    markers = markers+1

    # Mark the unknown region with zero
    markers[unknown==255] = 0

    # Watershed segementation 
    img_ws = cv2.cvtColor(img_ws, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(img_ws, markers)
    return markers



@magicgui(call_button='Read image',
          fn = {'label': 'Image file', 'tooltip': 'Enter file name'},
          write_dm3_metadata = {'label': 'Write metadata'},
          fn_dm3_metadata = {'label': 'Metadata file'}
          )
def read_image(fn: pathlib.Path='top_leftc.npy',
               write_dm3_metadata: bool=False,
               fn_dm3_metadata: pathlib.Path=None
               ) -> napari.types.LayerDataTuple:

    root, ext = os.path.splitext(fn); ft = ext.lower().lstrip('.')
    global FN_ROOT
    FN_ROOT = root
    if ft == 'npy':
        img = np.load(fn)
        if np.issubdtype(img.dtype, np.floating):
            img = ski.exposure.rescale_intensity(img)
    elif ft == 'dm3':
        img_dm3 = dm3.DM3(fn)
        img = ski.exposure.rescale_intensity(img_dm3.imagedata, out_range=np.float64)
        PXSIZE = img_dm3.pxsize[0]
        PXUNIT = img_dm3.pxsize[1].decode()
        if write_dm3_metadata:
            if not fn_dm3_metadata:
                fn_dm3_metadata = root + '.txt'
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
    assert img.ndim == 2 
    return (img, {'name':'Original', 'colormap': 'gray', 
                  'interpolation2d': 'spline36'}, 'image')



@magicgui(call_button='Equalize histogram', scrollable=False)
def equalize_hist(image: napari.layers.Image, 
       kernel_size: Tuple[int, int]=(8,8), 
       clip_limit: float=0.05, 
       nbins: int=256) -> napari.layers.Image :

    m, n = image.data.shape
    if isinstance(kernel_size, int):
        ks = (m//kernel_size, n//kernel_size)
    else:
        ks = (m//kernel_size[0], n//kernel_size[1])

    img_he = ski.exposure.equalize_adapthist(image.data, ks, clip_limit, nbins)
    return napari.layers.Image(img_he, name='HistEqualized', colormap='gray',
                        interpolation2d='spline36')



@magicgui(call_button='Denoise', scrollable=False, 
          eps={'widget_type': 'FloatSpinBox', 
               'value': 1e-3, 'min': 1e-6, 'max': 1e-2, 'step': 1e-6
               })
def denoise(image: napari.layers.Image, 
       weight: float=0.1, 
       max_num_iter: int=20, 
       eps: float=1.0e-4,
       ) -> napari.layers.Image :

    img_dn = ski.restoration.denoise_tv_bregman(image.data, weight, 
                max_num_iter, eps, isotropic=True)
    return napari.layers.Image(img_dn, name='Denoised', colormap='gray',
                        interpolation2d='spline36')



@magicgui(call_button='Binarize', fg_islow={'label': 'LIFG'})
def binarize(image: napari.layers.Image,
             fg_islow: bool=True) -> napari.types.LayerDataTuple:

    threshold = ski.filters.threshold_otsu(image.data, nbins=256)
    if fg_islow:
        img_bin = image.data < threshold
    else :
        img_bin = image.data > threshold
    return (img_bin, {'name':'Binarized', 'colormap': 'gray',
                  'interpolation2d': 'nearest'}, 'image')



@magicgui(call_button='Remove Objects')
def remove_objects(image: napari.layers.Image,
                   min_size: int=64,
                   labels: napari.layers.Labels=None,
                   shapes: napari.layers.Shapes=None
                   ) -> napari.types.LayerDataTuple:
    assert np.issubdtype(image.dtype, bool)
    if (labels is None) and (shapes is None):
        img = ski.morphology.remove_small_objects(image.data, min_size,
                                                connectivity=1)
    else:
        img = copy.deepcopy(image.data)
        if labels:
            print(labels.data.dtype, labels.data.shape)
            img[labels.data == 1 ] = False
        if shapes:
            for each in shapes.data:
                rr, cc = ski.draw.polygon(each[:,0], each[:,1],
                                                shape=image.data.shape)
                img[rr,cc] = False
    return (img, {'name':'Binarized.ObjectsRemoved', 'colormap': 'gray',
                  'interpolation2d': 'nearest'}, 'image')


@magicgui(call_button='Fill Holes')
def fill_holes(image: napari.layers.Image,
                 max_area: int=64,
                 labels: napari.layers.Labels=None,
                 shapes: napari.layers.Shapes=None
                 ) -> napari.types.LayerDataTuple:
    assert np.issubdtype(image.dtype, bool)
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
    return (img, {'name':'Binarized.HolesFilled', 'colormap': 'gray',
                  'interpolation2d': 'nearest'}, 'image')


@magicgui(call_button='Detect Overlap', image={'label': 'Image Layer'},
          rtola={'label': 'AreaTol'}, rtolp={'label': 'PerimTol'})
def detect_overlap(image: napari.layers.Image,
            rtola: float=0.3, rtolp: float=0.2
        ) -> napari.types.LayerDataTuple :

    labels_data = np.zeros(image.data.shape, dtype=np.uint8)
    img = ski.exposure.rescale_intensity(image.data, out_range=np.float64)
    img = ski.util.img_as_ubyte(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    for i in range(num_contours):
        contour_perim = cv2.arcLength(contours[i], True)
        contour_area = cv2.contourArea(contours[i])
        rect = cv2.minAreaRect(contours[i])
        w = rect[1][0]; h = rect[1][1]
        rect_perim = 2*(w+h); rect_area = w*h

        if math.isclose(rect_perim, contour_perim, rel_tol=rtolp) and \
                math.isclose(rect_area, contour_area, rel_tol=rtola):
            rect_verts = np.intp( cv2.boxPoints(rect) )
            rect_verts[:,[0,1]] = rect_verts[:,[1,0]]
            labels_data = cv2.drawContours(labels_data, contours, i, 1, cv2.FILLED)
        else:
            labels_data = cv2.drawContours(labels_data, contours, i, 2, cv2.FILLED)
    return (labels_data, {'name':'Markers'}, 'labels')



@magicgui(call_button='Segment', labels={'label': 'Labels Layer'})
def segment(labels: napari.layers.Labels) -> napari.types.LayerDataTuple:

    shapes_data = []
    if np.any(labels.data==1):
        img = np.zeros(labels.data.shape, dtype=np.uint8)
        img[labels.data == 1] = 255
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        nc = len(contours)
        for i in range(nc):
            rect = cv2.minAreaRect(contours[i])
            rect_verts = np.intp( cv2.boxPoints(rect) )
            rect_verts[:,[0,1]] = rect_verts[:,[1,0]]
            shapes_data.append(rect_verts)

    if np.any(labels.data==2) and np.any(labels.data==3):
        img = np.array(labels.data, dtype=np.uint8, copy=True)
        img[labels.data == 1] = 0
        markers = run_watershed(img)
        idregmax = markers.max()
        img[:,:] = 0
        for i in range(2, idregmax+1): #Background is 1
            img[markers==i] = 255
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(contours[0])
            rect_verts = np.intp( cv2.boxPoints(rect) )
            rect_verts[:,[0,1]] = rect_verts[:,[1,0]]
            shapes_data.append(rect_verts)
            img[:,:] = 0

    return (shapes_data, {'name':'NplBndry', 'shape_type': 'rectangle',
                          'edge_width': 2.0, 'edge_color': 'lime', 
                          'face_color': [0]*4}, 'shapes')



@magicgui(call_button='Calculate size', fn_out={'label': 'Output file'})
def calc_size(shapes: napari.layers.Shapes,
                fn_out: pathlib.Path=None,
                clear_shapes: bool=True
               ) -> None :
    #Calculate the length of the two sides of each rectangle
    #for each in shapes.data:
    nshapes = shapes.nshapes
    rect_sizes = np.zeros((nshapes,2))
    for i in range(nshapes):
        verts = shapes.data[i]
        d = [np.linalg.norm(verts[1,:] - verts[0,:]),
             np.linalg.norm(verts[2,:] - verts[0,:]),
             np.linalg.norm(verts[3,:] - verts[0,:])]
        d.sort()
        rect_sizes[i,0] = d[1]; rect_sizes[i,1] = d[0]

    rect_sizes *= PXSIZE
    if fn_out is None:
        fn_out = FN_ROOT + '.csv'
    with open(fn_out, 'w') as fh:
        fh.write(f'Unit, {PXUNIT}, \n')
        fh.write(f'length, height, area\n')
        for i in range(nshapes):
            fh.write( '%g, %g, %g\n'%(rect_sizes[i,0], rect_sizes[i,1],
                rect_sizes[i,0]*rect_sizes[i,1]) )

    if clear_shapes:
        shapes.data = []
    


widgets = {'Read image': read_image,
           'Equalize histogram': equalize_hist,
           'Denoise': denoise, 
           'Binarize': binarize,
           'Remove Objects': remove_objects,
           'Fill Holes': fill_holes,
           'Detect Overlap': detect_overlap,
           'Segment': segment,
           'Calculate size': calc_size}

viewer = napari.Viewer(order=(0,1))

for key, val in widgets.items():
    dw = viewer.window.add_dock_widget(val, name=key,
                            area='bottom', tabify=False, 
                            add_vertical_stretch=True,
                            menu=viewer.window.window_menu)
    dw._close_btn = False

# start the event loop and show the viewer
if __name__ == '__main__' :
    napari.run()
