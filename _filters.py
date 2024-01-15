#!/usr/bin/env python

'''
Widgets for filtering grayscale images.

'''
import numpy as np
import cv2
import skimage as ski
from magicgui import magicgui
from napari.layers import Image
from napari.types import LayerDataTuple
from _kuwahara import kuwahara

@magicgui(
    call_button='Run', 
    kernel_width = {'widget_type': 'SpinBox', 'step': 1},
    kernel_height = {'widget_type': 'SpinBox', 'step': 1},
    name={'label': 'Output image layer name'}
    )
def filter_mean(image: Image, 
        kernel_width: int=5, 
        kernel_height: int=0, 
        name: str='FilteredMean'   
       ) -> LayerDataTuple :

    """
    Smooths an image using the mean filter.  Wraps ``cv::blur()`` from OpenCV.
    See https://docs.opencv.org/4.8.0/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    kernel_width : int
        Width of the filter kernel. If zero, `kernel_height` must be > 0 and
        `kernel_width` will be set equal to `kernel_height`. Both `kernel_width`
        and `kernel_height` cannot be zero.
    kernel_height : int
        Height of the filter kernel. If zero, `kernel_width` must be > 0 and
        `kernel_height` will be set equal to `kernel_width`. Both `kernel_width`
        and `kernel_height` cannot be zero. 
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Filtered image with 64-bit float datatype

    """
    assert ( (kernel_width != 0) or (kernel_height != 0) )
    if kernel_width == 0:
        kernel_width = kernel_height
    if kernel_height == 0:
        kernel_height = kernel_width

    img = ski.util.img_as_ubyte(image.data)
    img_dst = cv2.blur(img, (kernel_width, kernel_height))
    img_dst = ski.util.img_as_float64(img_dst)
    return (img_dst, 
            {'name': name, 'metadata': image.metadata, 'colormap': 'gray',
             'interpolation2d': 'spline36'}, 'image')



@magicgui(
    call_button='Run', 
    ksize = {'widget_type': 'SpinBox', 'min': 3, 'step': 2},
    name={'label': 'Output image layer name'}
    )
def filter_median(image: Image, 
        ksize: int=5, 
        name: str='FilteredMedian'   
       ) -> LayerDataTuple :

    """
    Smooths an image using the median filter.  Wraps ``cv::medianBlur()`` from
    OpenCV. 
    See https://docs.opencv.org/4.8.0/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    ksize : int
        Aperture linear size; it must be odd and greater than 1.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Filtered image with 64-bit float datatype

    """
    assert ksize > 1
    img = ski.util.img_as_ubyte(image.data)
    img_dst = cv2.medianBlur(img, ksize)
    img_dst = ski.util.img_as_float64(img_dst)
    return (img_dst, 
            {'name': name, 'metadata': image.metadata,
             'colormap': 'gray', 'interpolation2d': 'spline36'},
            'image')



@magicgui(
    call_button='Run', 
    kernel_width = {'widget_type': 'SpinBox', 'min': 3, 'step': 2},
    kernel_height = {'widget_type': 'SpinBox'},
    name={'label': 'Output image layer name'}
    )
def filter_gaussian(image: Image, 
        kernel_width: int=5, 
        kernel_height: int=0, 
        sigma_x: float=0.0, 
        sigma_y: float=0.0, 
        name: str='FilteredGaussian'   
       ) -> LayerDataTuple :

    """
    Smooths an image using a Gaussian filter.  Wraps ``cv::GaussianBlur()`` from OpenCV.
    See https://docs.opencv.org/4.8.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    kernel_width : int
        Width of the filter kernel, must be odd. 
    kernel_height : int
        Height of the filter kernel, must be zero or odd. If zero,
        `kernel_width` must be > 0 and `kernel_height` will be set equal to
        `kernel_width`.
    sigma_x : float
        Gaussian kernel standard deviation in X direction. If zero, it is set
        equal to `kernel_width`.
    sigma_y : float
        Gaussian kernel standard deviation in Y direction. If zero, it is set
        equal to `kernel_height`.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Filtered image with 64-bit float datatype

    """
    if kernel_height == 0:
        kernel_height = kernel_width
    assert ( kernel_height % 2 == 1 )

    if sigma_x == 0:
        sigma_x = kernel_width
    if sigma_y == 0:
        sigma_y = kernel_height

    img = ski.util.img_as_ubyte(image.data)
    img_dst = np.zeros(image.data.size, dtype=np.uint8)
    img_dst = cv2.GaussianBlur(img, (kernel_width, kernel_height), sigma_x,
                               img_dst, sigma_y)
    img_dst = ski.util.img_as_float64(img_dst)
    return (img_dst, 
            {'name': name, 'metadata': image.metadata,
             'colormap': 'gray', 'interpolation2d': 'spline36'},
             'image')



@magicgui(
    call_button='Run', 
    weight={'widget_type': 'FloatSpinBox', 'step': 1e-3},
    eps={'widget_type': 'FloatSpinBox', 'step': 1e-5},
    name={'label': 'Output image layer name'}
    )
def denoise_tv_bregman(image: Image, 
        weight: float=0.5, 
        max_num_iter: int=100, 
        eps: float=0.001,
        name: str='DenoisedBregman'   
       ) -> LayerDataTuple :

    """
    Performs total variation denoising using split-Bregman optimization.
    Wraps ``denoise_tv_bregman()`` from scikit-image. 
    See https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_bregman

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    weight : float
        Denoising weight. The smaller the weight, the more denoising (at the
        expense of less similarity to image).
    max_num_iter : int
        Maximum number of iterations used for the optimization.
    eps : float
        Tolerance for stopping iterations, must be > 0.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Denoised image with 64-bit float datatype

    """
    img_dn = ski.restoration.denoise_tv_bregman(image.data, weight=weight,
                max_num_iter=max_num_iter, eps=eps)
    img_dn = ski.util.img_as_float64(img_dn)
    return (img_dn, 
            {'name': name, 'metadata': image.metadata,
             'colormap': 'gray', 'interpolation2d': 'spline36'},
             'image')



@magicgui(
    call_button='Run', 
    lamda = {'label': 'lambda'},
    name={'label': 'Output image layer name'}
    )
def denoise_tvl1(image: Image, 
        lamda: float=0.5, 
        niters: int=30, 
        name: str='DenoisedTVL1'   
       ) -> LayerDataTuple :

    """
    Performs denoising using a total variation based Primal-dual algorithm.
    Wraps ``cv::denoise_TVL1()`` from OpenCV. 
    See https://docs.opencv.org/4.8.0/d1/d79/group__photo__denoise.html#ga7602ed5ae17b7de40152b922227c4e4f

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    lamda : float
        Denoising weight. As it is enlarged, the smooth (blurred) images are
        treated more favorably than detailed (but maybe more noised) ones.
        Roughly speaking, as it becomes smaller, the result will be more blur
        but more severe outliers will be removed. 
    niters : int
        Number of iterations that the algorithm will run. Of course, as more
        iterations as better, but it is hard to quantitatively refine this
        statement, so just use the default and increase it if the results are
        poor.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Denoised image with 64-bit float datatype

    """
    img_dn = np.zeros(image.data.shape, dtype=np.uint8)
    img = ski.util.img_as_ubyte(image.data)
    cv2.denoise_TVL1([img], img_dn, lamda, niters) 
    img_dn = ski.util.img_as_float64(img_dn)
    return (img_dn, 
            {'name': name, 'metadata': image.metadata,
             'colormap': 'gray', 'interpolation2d': 'spline36'},
             'image')



@magicgui(
    call_button='Run', 
    template_window_size={'widget_type': 'SpinBox', 'min': 3, 'step': 2},
    search_window_size={'widget_type': 'SpinBox', 'min': 5, 'step': 2} , 
    name={'label': 'Output image layer name'}
    )
def denoise_nl_means(image: Image, 
        h: float=3.0,
        template_window_size: int=7, 
        search_window_size: int=21, 
        name: str='DenoisedNlMeans'   
        ) -> LayerDataTuple :
    """
    Performs denoising using Non-local Means Denoising algorithm
    (http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/) with several
    computational optimizations. Noise expected to be a gaussian white noise.
    Wraps ``cv::fastNlMeansDenoising()`` from OpenCV. 

    See https://docs.opencv.org/4.8.0/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    h : float
        Parameter regulating filter strength. Big `h` value perfectly removes
        noise but also removes image details, smaller `h` value preserves details
        but also preserves some noise.
    template_window_size : int
        Size in pixels of the template patch that is used to compute weights. Should be odd.
    search_window_size : int
        Size in pixels of the window that is used to compute weighted average
        for given pixel. Should be odd. Affects performance linearly: greater
        `search_window_size` will require greater denoising time.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Denoised image with 64-bit float datatype

    """
    img_dn = np.zeros(image.data.shape, dtype=np.uint8)
    img = ski.util.img_as_ubyte(image.data)
    img_dn = cv2.fastNlMeansDenoising(img, img_dn, h, template_window_size, 
                                      search_window_size)
    img_dn = ski.util.img_as_float64(img_dn)
    return (img_dn, 
            {'name': name, 'metadata': image.metadata,
             'colormap': 'gray', 'interpolation2d': 'spline36'},
             'image')



@magicgui(
    call_button='Run', 
    d={'widget_type': 'Slider', 'min': 0, 'max': 100, 'step': 1},
    sigma_color={'widget_type': 'FloatSlider', 'min': 0, 'max': 200, 'step': 1},
    sigma_space={'widget_type': 'FloatSlider', 'min': 0, 'max': 200, 'step': 1},
    name={'label': 'Output image layer name'}
    )
def denoise_bilateral(image: Image, 
        d: int=5, 
        sigma_color: float=25.0, 
        sigma_space: float=25.0,
        name: str='DenoisedBilateral'   
        ) -> LayerDataTuple :
    """
    Applies the bilateral filter to an image.  It is very slow compared to most
    filters. Wraps ``cv::bilateralFilter()`` from OpenCV.

    See https://docs.opencv.org/4.8.0/d4/d86/group__imgproc__filter.html#ga13a01048a8a200aab032ce86a9e7c7be

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    d : int
        Diameter of each pixel neighborhood that is used during filtering. If 
        `d <= 0`, it is computed from `sigma_space`. Large filters can be very
        slow.
    sigma_color : float
        Filter sigma in the color space. A larger value of the parameter means
        that farther colors within the pixel neighborhood (see `sigma_space`) will
        be mixed together, resulting in larger areas of semi-equal color. The 2
        `sigma` values can be set same. If they are small (< 10), the filter will
        not have much effect, whereas if they are large (> 150), they will have
        a very strong effect, making the image look "cartoonish".
    sigma_space : float
        Filter sigma in the coordinate space. A larger value of the parameter
        means that farther pixels will influence each other as long as their
        colors are close enough (see `sigma_color`). When `d>0`, it specifies the
        neighborhood size regardless of `sigma_space`. Otherwise, `d` is proportional
        to `sigma_space`.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Denoised image with 64-bit float datatype

    """
    img = ski.util.img_as_ubyte(image.data)
    img_dn = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    img_dn = ski.util.img_as_float64(img_dn)
    return (img_dn, 
            {'name': name, 'metadata': image.metadata,
             'colormap': 'gray', 'interpolation2d': 'spline36'},
             'image')



@magicgui(
    call_button='Run', 
    method={'widget_type': 'ComboBox', 'choices': ['mean', 'gaussian']},
    name={'label': 'Output image layer name'}
    )
def denoise_kuwahara(image: Image, 
        method: str='mean',
        radius: int=3,
        sigma: float=None,
        name: str='DenoisedKuwahara'   
        ) -> LayerDataTuple :
    """
    Performs denoising using Kuwahara algorithm.

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    method : str
        Method used to compute the pixel values. {'mean' | 'gaussian'}
    radius : float
        Window radius (`winsize = 2 * radius + 1`)
    sigma : float
        Used if metod is "gaussian", automatically computed when `None` 
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Denoised image with 64-bit float datatype

    """
    img_dn = kuwahara(image.data, method=method, radius=radius, sigma=sigma) 
    img_dn = ski.util.img_as_float64(img_dn)
    return (img_dn, 
            {'name': name, 'metadata': image.metadata,
             'colormap': 'gray', 'interpolation2d': 'spline36'},
            'image')



@magicgui(
    call_button='Run', 
    amount = {'widget_type': 'FloatSpinBox', 'min': -100.0},
    name={'label': 'Output image layer name'}
    )
def filter_unsharp_mask(image: Image, 
        radius: float=1.0, 
        amount: float=1.0, 
        name: str='FilteredUnsharp'   
       ) -> LayerDataTuple :

    """
    Sharpens an image using the unsharp_mask filter.  Wraps ``unsharp_mask()`` from scikit-image.
    See https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.unsharp_mask

    Parameters
    ----------
    image : napari.layers.Image
        Input grayscale image
    radius : float
        Blurring radius. Note that 0 radius means no blurring.
    amount : float
        The details will be amplified with this factor. The factor could be 0 or
        negative. Typically, it is a small positive number, e.g. 1.0.
    name : str
        Name of output napari image layer

    Returns
    -------
    napari.types.LayerDataTuple
        Filtered image with 64-bit float datatype

    """
    img_dst = ski.filters.unsharp_mask(image.data, radius=radius, amount=amount)
    img_dst = ski.util.img_as_float64(img_dst)
    return (img_dst, 
            {'name': name, 'metadata': image.metadata,
             'colormap': 'gray', 'interpolation2d': 'spline36'},
            'image')



filter_widgets = [(filter_mean, 'Mean'),
                  (filter_median, 'Median'),
                  (filter_gaussian, 'Gaussian'),
                  (filter_unsharp_mask, 'Sharpen')
                 ]

denoiser_widgets = [
                    (denoise_tv_bregman, 'TV Bregman'),
                    (denoise_tvl1, 'TVL1'),
                    (denoise_nl_means, 'Nonlocal Means'),
                    (denoise_bilateral, 'Bilateral'),
                    (denoise_kuwahara, 'Kuwahara')
                   ]
