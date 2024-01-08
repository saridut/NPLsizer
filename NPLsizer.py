#!/usr/bin/env python

'''
Driver script for calculating nanoplatelet size from an image.

'''

import napari
from _io_ops import io_op_widgets
from _histogram import hist_widgets
from _filters import filter_widgets, denoiser_widgets
from _thresholders import threshold_widgets
from _morph_ops import morph_op_widgets
from _segmentation import segmentation_widgets


viewer = napari.Viewer(order=(0,1))
# Napari Window menu
menu_nw = viewer.window.window_menu


# Add I/O widgets
for each in io_op_widgets:
    dw = viewer.window.add_dock_widget(each[0], name=each[1],
        allowed_areas=['right'], menu=menu_nw)
    dw._close_btn = False; dw.setVisible(False); dw.setFloating(True)


# Add histogram widgets
menu_hist = menu_nw.addMenu('Histogram')
for each in hist_widgets:
    dw = viewer.window.add_dock_widget(each[0], name=each[1], 
        allowed_areas=['right'], menu=menu_hist)
    dw._close_btn = False; dw.setVisible(False); dw.setFloating(True)


# Add filter widgets
menu_filter = menu_nw.addMenu('Filter')
for each in filter_widgets:
    dw = viewer.window.add_dock_widget(each[0], name=each[1], allowed_areas=['right'],
           menu=menu_filter)
    dw._close_btn = False; dw.setVisible(False); dw.setFloating(True)


# Add denoiser widgets
menu_denoise = menu_nw.addMenu('Denoise')
for each in denoiser_widgets:
    dw = viewer.window.add_dock_widget(each[0], name=each[1], allowed_areas=['right'],
           menu=menu_denoise)
    dw._close_btn = False; dw.setVisible(False); dw.setFloating(True)


# Add threshold widgets
menu_threshold = menu_nw.addMenu('Threshold')
for each in threshold_widgets:
    dw = viewer.window.add_dock_widget(each[0], name=each[1], allowed_areas=['right'],
           menu=menu_threshold)
    dw._close_btn = False; dw.setVisible(False); dw.setFloating(True)


# Add morphological operations widgets
menu_morph_op = menu_nw.addMenu('Morphology')
for each in morph_op_widgets:
    dw = viewer.window.add_dock_widget(each[0], name=each[1], allowed_areas=['right'], 
           menu=menu_morph_op)
    dw._close_btn = False; dw.setVisible(False); dw.setFloating(True)


# Add segmentation widgets
menu_segment = menu_nw.addMenu('Segmentation')
for each in segmentation_widgets:
    dw = viewer.window.add_dock_widget(each[0], name=each[1], allowed_areas=['right'], 
           menu=menu_segment)
    dw._close_btn = False; dw.setVisible(False); dw.setFloating(True)


# start the event loop and show the viewer
if __name__ == '__main__' :
    napari.run()
