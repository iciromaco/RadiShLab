# RadiShLab
![radishla128b](https://user-images.githubusercontent.com/24559785/73605022-e488d000-4590-11ea-9530-95b6267f634a.png)  
Radish Shape Analysis Laboratory

## Description

**RadiShLab** is a python library for shape analysis of radishes.

  - Generatation of silhouette images from radish photos
  - Silhouette shape description
  - Measurement taking into account the radish body bending

# New!

  - Rewritten code for OPENCV 4.X

====
## Demo
[![](https://img.youtube.com/vi/geLT5e6Tkqg/0.jpg)](https://www.youtube.com/watch?v=geLT5e6Tkqg)

## Requirement
- ### python and jupyter notebook execution environment
（Recomendation1)  [**Google Colaboratory**](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja). 

  The easiest way is to use Google Colaboratory. This is a cloud execution environment of the jupyter notebook
   provided by Google. In the environment, the necessary libraries are already installed. 
   You can use expensive GPUs and TPUs for machine learning by free. However, since Colaboratory cannot 
   open a window on the local PC's monitor, it cannot execute programs with GUIs.

（Recomendation12） [**Anaconda**](https://www.anaconda.com/python) (python programming 具体的には，Environment)

  Any other python distributions are OK if **jupyter notebook**,**numpy**, **matplotlib** and **opencv** are in it．

python Anaconda，and OpenCV - Image Processing Library are updated very often.

It is not necessary to keep up with the latest, but to save the trouble of specifying the version, 
confusion of version differences for each user, the trouble of updating the library, etc., 
and because the explanation is easier if the environment is unified ,
 (Recommended 1) will be based on running at Google Colaboratory.
 
If you use a program that uses a GUI, you will also need to install the (Recommended 2) Anaconda environment. 
In that case, there is no need to use Google Colaboratory, 
but since Google virtual machines have higher performance than most PCs, 
process them in Colaboratory except when using the GUI, 
and work only on the local PC's Anaconda for the GUI. 
You may want to do that.

## Install Anaconda and OpenCV 
### Install Anaconda 
Download the installer for your OS (Windows, macOS, Linux) from the above site and install it.

All major libraries required for scientific calculations are installed by default.

### Creating a virtual environment
Python has complicated dependencies between libraries and often requires out-of-date libraries. 
Since the required version in one project is different from the required version in another project, 
it is common to create a virtual environment for each project and switch between them.

Since the articles on the web are written by persons who are very familiar with computers, 
it describes how to create and switch virtual environments by typing commands with CUI.
But it is preferred that you use Anaconda Navigator to create and switch virtual environments.
Anaconda Navigator makes it easy for people who are not used to CUI operations to create virtual environments and switch them.

Specifically, you can create a new environment by opening Navigator's Environment and clicking the + mark.

## Update Anaconda

Please update the local environment regularly so that the version of Google Colaboratory and the library do not differ.
- Update Anaconda itself to the latest version
- Update all libraries to the latest version

### Update Anaconda itself to the latest version
``` $ conda update -n base cond```
### Update all libraries to the latest version
``` $ conda update --all```

## Install

Just download the zip of this repository and unzip it to any directory.

## Usage

Notebooks are provided for each functions. 
In the terminal, start **jupyter notebook** in the directory of the notebook you want to try, 
open the notebook in your browser and execute the program. 
It is a good idea to run it on a duplicate of the notebook to avoid breaking it.

If you run them on Google Colaboratory, upload and run the notebook.

## How to
### Using GUI
- [Google Colaboratory]()

## Author
- Seiichiro Dan： Osaka Gakuin Univ.
- Yumiko Dan, Yasuko Yoshida： Kobe Univ.

## Related Links
- [SHAPE](http://lbm.ab.a.u-tokyo.ac.jp/~iwata/shape/index.html)