# NPLsizer
Python code to semi-automatically calculate the size of rectangular nanoplatelets from microscopy images.

# Installation
## Prerequisites
[Python >= 3.8](https://www.python.org/downloads/) is required for installation,
along with the following Python packages. 

- [numpy (1.24)](https://numpy.org/)
- [scipy (1.9)](https://scipy.org/)
- [pillow (10.1)](https://python-pillow.org/)
- [PyQt (5.15)](https://pypi.org/project/PyQt5/)
- [magicgui (0.8)](https://pyapp-kit.github.io/magicgui/)
- [napari (0.4)](https://napari.org/stable/)
- [opencv (4.8)](https://opencv.org/)
- [scikit-image (0.20)](https://scikit-image.org/)
- [pyDM3reader (1.5)](https://github.com/piraynal/pyDM3reader)

Please note that I have tested the code only with Python 3.8 with the specific
versions of the dependencies listed above inside parentheses, so make sure all
the dependencies work if you are using a later version.

## Installation using Conda
I recommend using the [Conda Package Manager](https://conda.io/projects/conda/en/latest/user-guide/index.html) to install all the prerequisites. Make sure to use 
`conda >= 22.11`.

1. After installing `conda`, it is recommended to use a clean virtual
environment for installing all the prerequisites. The code below creates a
new environment `myenv` with `python 3.8`.

```bash
    conda create -y -n myenv -c conda-forge python=3.8
    conda activate myenv
```

2. [*Optional*] Sometime `conda` has difficulty in locating the appropriate set
   of compatible packages. It may help to change the default solver to
`libmamba`.

```bash
    conda update -n base conda
    conda install -n base conda-libmamba-solver
    conda config --set solver libmamba
```

3. Install most of the prerequisites. `scikit-image` and `pillow` should be
   automatically installed by `conda` as a dependency of some of these packages.

```bash
    conda install -c conda-forge --override-channels numpy scipy napari pyqt magicgui opencv
```

4. Download a `zip` file of `pyDM3reader` from the website listed above and save
   it in your working directory.

```bash
    pip install pyDM3reader-1.5.zip
```

5. Update everything. 

```bash
    conda update --all
```
