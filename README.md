# RadiShLab
![radishla128b](https://user-images.githubusercontent.com/24559785/73605022-e488d000-4590-11ea-9530-95b6267f634a.png)  
Radish Shape Analysis Laboratory

## Description

**RadiShLab** は プログラミング言語 python で記述されたダイコンの形状解析のためのライブラリです.

  - 原画像からのシルエット画像を生成
  - シルエットの形状記述
  - 曲がりを考慮した計測

**RadiShLab** is a python library for shape analysis of radishes.

  - Generatation of silhouette images from radish photos
  - Silhouette shape description
  - Measurement taking into account the radish body bending

# New!

  - OPENCV 4.X 用にコードを書き換え
  - Rewritten code for OPENCV 4.X

====
## Demo
[![](https://img.youtube.com/vi/geLT5e6Tkqg/0.jpg)](https://www.youtube.com/watch?v=geLT5e6Tkqg)

## Requirement
### python と jupyter notebook が実行できる環境
-（推奨1） [**Anaconda**](https://www.anaconda.com/python) (python プログラミング開発環境)

  **jupyter notebook**,**numpy**, **matplotlib** and **opencv** などが含まれているなら他のディストリビューションでも構いません．
  
- （推奨2)  [**Google Colaboratory**](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja). 

  一番楽なのは Google Colaboratory を使うことです．これは Google が提供してくれる jupyter notebook の実況環境であり，必要なライブラリがすべて含まれています．ただし，colaboratory ではGUI を使うプログラムを実行できません．
  
## Install

このリポジトリのzipをダウンロードして任意のディレクトリに解凍するだけです．

## Usage

機能ごとにノートブックにしてあります．ターミナルで試したいノートブックのあるディレクトリで jupyter notebook を起動し，ブラウザでノートブックを開いてプログラムを実行してください．ノートブックを壊してしまわないよう，複製上で実行するのがよいでしょう．

Google Colaboratory で実行する場合は，ノートブックをアップロードして実行してください．


====
English Description and Instruction
====
## Requirement
### python and jupyter notebook execution environment
- (recomended 1) [**Anaconda**](https://www.anaconda.com/python) (python programming environment)

  Any other python distribution may replace Anaconda if it contains libraries **jupyter notebook**,**numpy**, **matplotlib** and **opencv**.
  
  The easiest way to try RadishLab is to run it on [**Google Colaboratory**](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja). However, GUI programs cannot be used in colaboratory.


## Install

Download zip file of this repository and expand it to any directory.

## Usage

Open terminal and start the jupyter notebook in the directory where the notebook you want to try it, open the notebook in your browser and run the program. It's a good idea to run it on a duplicate of the notebook you want to try.

If you want to run on Google Colaboratory, upload the notebook first and run it.

## Author
- Seiichiro Dan, Osaka Gakuin Univ.
- Yasuko Yoshida, Kobe Univ.

## Related Links
- [SHAPE](http://lbm.ab.a.u-tokyo.ac.jp/~iwata/shape/index.html)

