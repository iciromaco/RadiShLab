# RadiShLab
![logo](https://user-images.githubusercontent.com/24559785/76580094-066f5e00-6512-11ea-9963-66e3bb6d2cbc.png)
Radish Shape Analysis Laboratory

## Description

**RadiShLab** is a python library for shape analysis of radishes.

  - Generatation of silhouette images from radish photos  
  - Silhouette shape description  
  - Measurement taking into account the radish body bending  

# New!

  - Rewritten code for OPENCV 4.X

====

<div align="center">
<img src="https://user-images.githubusercontent.com/24559785/75554908-28a5ae00-5a33-11ea-9b1e-ce737aa6d440.gif" width=512> </img>

<!-- <video src="https://www.youtube.com/watch?v=geLT5e6Tkqg" width=512 align='center'><img src="https://img.youtube.com/vi/geLT5e6Tkqg/0.jpg" width=512></video>） -->

![ddd](https://user-images.githubusercontent.com/24559785/76141224-10004e00-605a-11ea-9c36-a27888906e22.png)

</div>

# :high_brightness:Lazy Way:high_brightness:   For those who want to see what it's like , or who are not willing to install the development environment</H1>

Try it on Google Colaboratory.  
- No need to install development environment  
- Only need a browser, you can run it on your smartphone or tablet.  

<img src="https://user-images.githubusercontent.com/24559785/76582358-5b62a280-6519-11ea-87ad-5b7a3ef90f24.gif" width=300>

1. Download and unzip the zip file of this repository. Save it directly under Google Drive named 'RadiShLab'.  
2. Log in to [Google Laboratory] (https://colab.research.google.com/)  
3. Open Notebook > Search GitHub for *iciromaco*  
4. Open a notebook file in the *iciromaco/RadiShLab* repository  

All you have to do is follow the explanation and run the code.

You'll need permission to access the files you've saved to Google Drive.

<img src="https://user-images.githubusercontent.com/24559785/76583297-315eaf80-651c-11ea-8ab7-ef47db3aa9bb.gif" width=500>

# Developer's way
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
``` $ conda update -n base conda```
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

-  [Local Runtime](https://research.google.com/colaboratory/local-runtimes.html)

## Author
- Seiichiro Dan： Osaka Gakuin Univ.
- Yumiko Dan, Yasuko Yoshida： Kobe Univ.

## Related Links
- [SHAPE](http://lbm.ab.a.u-tokyo.ac.jp/~iwata/shape/index.html)
