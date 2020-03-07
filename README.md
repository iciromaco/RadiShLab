# RadiShLab (4th)

![radishla128b](https://user-images.githubusercontent.com/24559785/73605022-e488d000-4590-11ea-9530-95b6267f634a.png)  
Radish Shape Analysis Laboratory

[English](https://github.com/iciromaco/RadiShLab/blob/master/README-Eng.md)

## Description

**RadiShLab** は プログラミング言語 python で記述されたダイコンの形状解析のためのライブラリです.

  - 原画像からのシルエット画像を生成
  - シルエットの形状記述
  - 曲がりを考慮した計測

# New!

  - OPENCV 4.X 用にコードを書き換え

====
## Demo

<div align="center">
<img src="https://user-images.githubusercontent.com/24559785/75554908-28a5ae00-5a33-11ea-9b1e-ce737aa6d440.gif" width=512> </img>
<video src="https://www.youtube.com/watch?v=geLT5e6Tkqg" width=512 align='center'><img src="https://img.youtube.com/vi/geLT5e6Tkqg/0.jpg" width=512></video>

![ddd](https://user-images.githubusercontent.com/24559785/76141224-10004e00-605a-11ea-9c36-a27888906e22.png)

</div>

## Requirement
- ### python と jupyter notebook が実行できる環境
（推奨1)  [**Google Colaboratory**](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja). 

  一番楽なのは Google Colaboratory を使うことです．これは Google がクラウドで提供してくれる jupyter notebook の実行環境であり，必要なライブラリはインストール済みです．機械学習用の高価なGPUやTPUを使うこともできます．ただし，Colaboratory はローカルPCにウィンドウを開けないので，GUI を使うプログラムを実行できません．

（推奨2） [**Anaconda**](https://www.anaconda.com/python) (python プログラミング開発環境)

  **jupyter notebook**,**numpy**, **matplotlib** and **opencv** などが含まれているなら他のディストリビューションでも構いません．

python や Anaconda，そして 画像処理ライブラリ OpenCV は非常に頻繁にバージョンが更新されていきます．

最新を追いかける必要はないわけですが，インストールした時期によってバージョンがまちまちになるのを避けるのと，
バージョン指定の面倒とライブラリの更新の手間を省くためと，環境が統一されていたら説明が楽だという理由で，（推奨１）のGoogle Colaboratory で実行することを基本にすることにします．

GUIを使うプログラムを使う人は（推奨２）の Anaconda 環境もインストールすることになります．その場合は Google Colaboratory を使う必要はありませんが，Google の仮想マシンはたいていのPCより性能が上ですので，GUIを使うとき以外は Colaboratory で処理し，GUIの必要な作業だけローカルPCの Anaconda で作業するとよいでしょう．

## Anaconda と OpenCV のインストール
ほとんどの処理は，Google Colaboratory でできますので，特に何もインストールする必要はありませんが，GUIを使ったインタラクティブな処理がしたい場合は，ローカルPCに Anaconda と OpenCV をインストールしてください．

### 1. Anaconda のインストール
上記サイトから自分のOS（Windows，macOS, Linux）に合ったインストーラをダウンロードしてインストールしてください．

標準で科学技術計算に必要な主だったライブラリはすべてインストールされます．

### 2. 仮想環境の作成
必須ではありませんが，専用の仮想環境を作成することをお勧めします．

python はライブラリ間の依存関係が複雑でしばしば最新でないライブラリが要求されたりします．あるプロジェクトで必要なバージョンと別のプロジェクトで必要なバージョンが違っていたりするので，プロジェクトごとに仮想環境を作って切り替えて使うのが普通です．

ネットの記事は計算機に詳しい人が書いているので CUI でコマンドを打って仮想環境を作ったり，切り替えたりする方法が書かれていますが，Anacondaに含まれている Anaconda Navigator を使えば，GUIで仮想環境を作成したり切り替えたりできるので，CUI操作に慣れていない人も取っつきやすいでしょう．

1. Anaconda Navigator を起動
2. Environments > +Create をクリック
3. Nameを適当につけ，Python 3.7 を選んで Create
4. Home > Jupyter Notebook > Install

### 3. ライブラリのインストール
#### OpenCV の導入

画像処理には，画像処理ライブラリ OpenCV を使います．Google Colaboratory には，つねにその最新に近いバージョンがインストールされています．OpenCV は Anacondaの標準ライブラリではないので，Google Colaboratoryと同じバージョンをインストールしておきましょう．OpenCVは頻繁にバージョンアップされ，ときどき仕様が変わるので注意が必要です．

1. Google Colaboratory に使われている OpenCV のバージョンを確認する．   
![230624](https://user-images.githubusercontent.com/24559785/73751807-bacdd580-4757-11ea-9bfd-cc0d698ad277.png)
2. Anaconda Navigator から仮想環境のターミナルを起動   
 ![231152](https://user-images.githubusercontent.com/24559785/73752501-fddc7880-4758-11ea-87d8-8a3e3a17d50e.png)
3. ターミナルで次のコマンドを実行   
```
conda install -c conda-forge opencv=4.1.2

Proceed ([y]/n)? y
```

### kivy
```
conda install -c conda-forge kivy
```
日本語が使いたい場合は、
```
pip install japanize-kivy
```

このプロジェクトではGUIプログラムには [kivy](https://kivy.org) を使っています。
python 用の GUI のライブラリにはまだこれこそがスタンダードだ、というものがありませんがので、今後もっと使いやすいものが出てきたら変更するかもしれません。

#### core なライブラリ
Anaconda で仮想環境を作った場合、ライブラリは最小限しか組み込まれていないので、次のライブラリも追加インストールしてください。
- numpy
- Pillw (PIL)
- matplotlib
- seaborn
- jupyter
- sympy 
- scikit-learn 
- pandas

```
conda install Pillow numpy jupyter matplotlib seaborn
```

#### その他
予定
- tensorflow
- keras

## Anaconda の更新

ローカルな環境はGoogle Colaboratory とライブラリのバージョンが乖離しないよう，定期的に更新してください．
- Anaconda のベースを最新にバージョンアップ
- すべての標準ライブラリを最新にバージョンアップ
が必要です．

### Anaconda のベースを最新にバージョンアップ
``` $ conda update -n base conda```
### すべてのライブラリをバージョンアップ
``` $ conda update --all```

## Install

このリポジトリのzipをダウンロードして任意のディレクトリに解凍するだけです．

## Usage

機能ごとにノートブックにしてあります．ターミナルで試したいノートブックのあるディレクトリで jupyter notebook を起動し，ブラウザでノートブックを開いてプログラムを実行してください．ノートブックを壊してしまわないよう，複製上で実行するのがよいでしょう．

Google Colaboratory で実行する場合は，ノートブックをアップロードして実行してください．

## How to
### GUI を使うプログラムの実行
Google Colaboratory のプログラムは、通常はクラウド上の仮想Linuxマシンで実行されるため、ローカルPCのディスプレイリソースを使うプログラム、つまりウィンドウを開くようなプログラムが実行できませんが、ローカルPC上のランタイムと接続すればそのようなプログラムも実行可能です。

-  [ローカル ランタイム](https://research.google.com/colaboratory/local-runtimes.html?hl=ja)


## Author
- Seiichiro Dan： Osaka Gakuin Univ.
- Yumiko Dan, Yasuko Yoshida： Kobe Univ.

==

## Related Links
- [SHAPE](http://lbm.ab.a.u-tokyo.ac.jp/~iwata/shape/index.html)
- [Anacondaのインストール(Win,Mac,Linux)](https://www.python.jp/install/anaconda/index.html)

