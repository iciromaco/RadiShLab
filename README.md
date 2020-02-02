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
- ### python と jupyter notebook が実行できる環境
（推奨1)  [**Google Colaboratory**](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja). 

  一番楽なのは Google Colaboratory を使うことです．これは Google がクラウドで提供してくれる jupyter notebook の実況環境であり，必要なライブラリがすべて含まれています．あなたは何もインストールする必要はありません．ただし，Colaboratory はローカルPCにウィンドウを開けないので，GUI を使うプログラムを実行できません．

（推奨2） [**Anaconda**](https://www.anaconda.com/python) (python プログラミング開発環境)

  **jupyter notebook**,**numpy**, **matplotlib** and **opencv** などが含まれているなら他のディストリビューションでも構いません．

python や Anaconda，そして 画像処理ライブラリ OpenCV は非常に頻繁にバージョンが更新されていきます．

最新を追いかける必要はないわけですが，インストールした時期によってバージョンがまちまちになるのを避けるのと，
バージョン指定の面倒とライブラリの更新の手間を省くためと，実行環境を統一する目的で，（推奨１）のGoogle Colaboratory で実行することを基本にすることにします．

GUIを使うプログラムを使う人は（推奨２）の Anaconda 環境もインストールすることになります．その場合は Google Colaboratory を使う必要はありませんが，Google の仮想マシンはたいていのPCより性能が上ですので，無駄ではないと思います．

## Anaconda と OpenCV のインストール
### Anaconda のインストール
上記サイトから自分のOS（Windows，macOS, Linux）に合ったインストーラをダウンロードしてインストールしてください．

標準で科学技術計算に必要な主だったライブラリはすべてインストールされます．

### 仮想環境の作成
python はライブラリ間の依存関係が複雑でしばしば最新でないライブラリが要求されたりします．あるプロジェクトで必要なバージョンと別のプロジェクトで必要なバージョンが違っていたりするので，プロジェクトごとに仮想環境を作って切り替えて使うのが普通です．

ネットの記事は計算機に詳しい人が書いているので CUI でコマンドを打って仮想環境を作ったり，切り替えたりする方法が書かれていますが，Anacondaに含まれている Anaconda Navigator を使えば，GUIで仮想環境を作成したり切り替えたりできるので，CUI操作に慣れていない人も取っつきやすいでしょう．

具体的には，Environment というのが仮想環境のことで，＋マークをクリックすることで新しい環境を作れます．


## Anaconda の更新

ローカルな環境はGoogle Colaboratory とライブラリのバージョンが乖離しないよう，定期的に更新してください．
- Anaconda のベースを最新にバージョンアップ
- すべての標準ライブラリを最新にバージョンアップ
が必要です．

### Anaconda のベースを最新にバージョンアップ
``` $ conda update -n base cond```
### すべてのライブラリをバージョンアップ
``` $ conda update --all```

## Install

このリポジトリのzipをダウンロードして任意のディレクトリに解凍するだけです．

## Usage

機能ごとにノートブックにしてあります．ターミナルで試したいノートブックのあるディレクトリで jupyter notebook を起動し，ブラウザでノートブックを開いてプログラムを実行してください．ノートブックを壊してしまわないよう，複製上で実行するのがよいでしょう．

Google Colaboratory で実行する場合は，ノートブックをアップロードして実行してください．

## How to
### GUI を使うプログラムの実行
- [Google Colaboratory を自分のマシンで走らせる]()

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

## How to
### 

## Author
- Seiichiro Dan, Osaka Gakuin Univ.
- Yasuko Yoshida, Kobe Univ.

## Related Links
- [SHAPE](http://lbm.ab.a.u-tokyo.ac.jp/~iwata/shape/index.html)

