#!/usr/bin/env python

"""
Widgets for segmenting binary image of regtangular nanoplatelets.

"""

import math
import numpy as np
import cv2
import skimage as ski
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes
from napari.types import LayerDataTuple


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



@magicgui(call_button='Run',
          rtola={'label': 'Area tolerance'},
          rtolp={'label': 'Perimeter tolerance'})
def detect_overlap(image: Image,
            rtola: float=0.3, 
            rtolp: float=0.2
            ) -> LayerDataTuple :
    """
    Labels connected and unconnected shapes.

    Parameters
    ----------
    rtola : float
        Area tolerance
    rtolp : float
        Perimeter tolerance

    Returns
    -------
    napari.layers.Labels
        A labels layer named `Markers`.

    """

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
    return (labels_data, {'name':'Markers', 'metadata': image.metadata}, 'labels')



@magicgui(call_button='Run',
          name={'label': 'Output shapes layer name'}
          )
def segment(labels: Labels,
            edge_color: str='lime',
            edge_width: float=2.0,
            name='NplBndry'
            ) -> LayerDataTuple:
    """
    Performs segmentation using watershed algorithm.

    Parameters
    ----------
    labels : napari.layers.Labels
        Labels layer.
    edge_color : str
        Can be any color name recognized by VisPy or hex value (in lower case) 
        if starting with #.
    edge_width : float
        Thickness of lines and edges.
    name : str
        Name of the output shapes layer

    Returns
    -------
    napari.layers.Shapes
        A shapes layer.

    """

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

    return (shapes_data, {'name': name, 'metadata': labels.metadata,
                          'shape_type': 'rectangle', 'edge_width': edge_width,
                          'edge_color': edge_color, 'face_color': [0]*4},
                          'shapes')



segmentation_widgets = [(detect_overlap, 'Label'),
                        (segment, 'Segment')
                       ]
