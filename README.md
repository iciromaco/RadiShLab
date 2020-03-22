# RadiShLab (4th)

![logo](https://user-images.githubusercontent.com/24559785/76580094-066f5e00-6512-11ea-9963-66e3bb6d2cbc.png)
Radish Shape Analysis Laboratory

[English](https://github.com/iciromaco/RadiShLab/blob/master/README-Eng.md)

## Description

- **RadiShLab** は プログラミング言語 python で記述されたダイコンの形状解析のためのプログラム集です。
- jupyter notebook 形式の実行できるノートを公開します。（N001,N002〜）

  - 原画像からのシルエット画像を生成
  - シルエットの形状記述（ベジエ曲線、フーリエ記述子）
  - 曲がりを考慮した計測（ベジエの応用）

----

# New!

  - OPENCV 4.X 用にコードを書き換え

----

## Preview
### N003 セグメンテーション
<div align="center">
<img src="https://user-images.githubusercontent.com/24559785/75554908-28a5ae00-5a33-11ea-9b1e-ce737aa6d440.gif" width=512> </img>
</div>

### N005 輪郭線分割
<div align="center">
<!-- <video src="https://www.youtube.com/watch?v=geLT5e6Tkqg" width=512 align='center'><img src="https://img.youtube.com/vi/geLT5e6Tkqg/0.jpg" width=512></video>） -->

<img src="https://user-images.githubusercontent.com/24559785/76141224-10004e00-605a-11ea-9c36-a27888906e22.png">
</div>

### N006 ベジエ曲線のあてはめ
<div align="center">
<img src="https://user-images.githubusercontent.com/24559785/77037479-cc5ff980-69f4-11ea-9bc8-1087dbd5b50d.png" width=240><img src="https://user-images.githubusercontent.com/24559785/77037483-cec25380-69f4-11ea-965f-1588d982d4f5.png" width=240>
</div>

### N007 曲率解析
<div align="center">
<img src="https://user-images.githubusercontent.com/24559785/77162234-d5cd8c80-6aee-11ea-8ccd-b62609ba2425.png" width=480>
</div>

### N008 中心軸の記述
<div align="center">
<img src="https://user-images.githubusercontent.com/24559785/77162115-8f782d80-6aee-11ea-8452-1b220e8adaef.png" width=480>
<img src="https://user-images.githubusercontent.com/24559785/77162548-8dfb3500-6aef-11ea-8b48-42697d2dd459.png" width=480>
</div>

#### N009補正形状の生成と計測
<div align="center">
<img src="https://user-images.githubusercontent.com/24559785/77229405-1d2c4980-6bd1-11ea-85a8-95e35b0feefb.png" width=480>

<img src="https://user-images.githubusercontent.com/24559785/77253072-85dff880-6c9b-11ea-95d4-83857770259f.png" width=200><img src="https://user-images.githubusercontent.com/24559785/77253074-86788f00-6c9b-11ea-81b2-03c4be000fa6.png" width=200><img src="https://user-images.githubusercontent.com/24559785/77253071-84aecb80-6c9b-11ea-89e1-bc02b5719a1b.png" width=200></div>
----

# :high_brightness:Lazy Way:high_brightness: 
>とりあえずどんなものか見てみたい方、開発環境のインストールなしで済ませたい方へ

Google Colaboratory で試してみてください。
- 開発環境のインストール不要
- ブラウザさえあれば実行可能、スマホやタブレットでも実行可能
<img src="https://user-images.githubusercontent.com/24559785/76582358-5b62a280-6519-11ea-87ad-5b7a3ef90f24.gif" width=300>

1. このリポジトリの zip をダウンロードし、解凍。フォルダ名を「RadiShLab」に書き換え、自分の Google Drive 直下に保存する。  
![repositoryZipDownload](https://user-images.githubusercontent.com/24559785/76692365-dacab000-6698-11ea-943b-dcb2d3fcd4e9.png)
2. [Google Colaboratory](https://colab.research.google.com/)にログイン
3. ノートブックを開く　＞　GitHub で、 "**iciromaco**" を検索
4. iciromaco/RadiShLab のリポジトリにあるノートブックファイルを開く

あとは説明にしたがって上から実行していくだけです。

Google ドライブに保存したファイルにアクセスしますので、許可が必要になります。

<img src="https://user-images.githubusercontent.com/24559785/76583297-315eaf80-651c-11ea-8ab7-ef47db3aa9bb.gif" width=500>

----
# Developer's way
> 自分のパソコン上に開発環境を構築して実行してみたい方、GUIツールも使いたい方

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
[上記サイト](https://www.anaconda.com/python)から自分のOS（Windows，macOS, Linux）に合ったインストーラをダウンロードしてインストールしてください．

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
GUI(kivy)で日本語が使いたい場合は、
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
conda install Pillow numpy jupyter matplotlib seaborn sympy scikit-learn pandas
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

____

## RadiShLab 自体のインストール
![repositoryZipDownload](https://user-images.githubusercontent.com/24559785/76692365-dacab000-6698-11ea-943b-dcb2d3fcd4e9.png)

このリポジトリのzipをダウンロードして任意のディレクトリに解凍するだけです．

## Usage

機能ごとにノートブックにしてあります．ターミナルで試したいノートブックのあるディレクトリで jupyter notebook を起動し，ブラウザでノートブックを開いてプログラムを実行してください．ノートブックを壊してしまわないよう，複製上で実行するのがよいでしょう．

Google Colaboratory で実行する場合は，ノートブックをアップロードして実行してください．

## How to
### GUI を使うプログラムの実行
Google Colaboratory のプログラムは、通常はクラウド上の仮想Linuxマシンで実行されるため、ローカルPCのディスプレイリソースを使うプログラム、つまりウィンドウを開くようなプログラムが実行できませんが、ローカルPC上のランタイムと接続すればそのようなプログラムも実行可能です。

-  [ローカル ランタイム](https://research.google.com/colaboratory/local-runtimes.html?hl=ja)

----

## &copy; 2017-2020 
- Seiichiro Dan： Osaka Gakuin Univ.
- Yumiko Dan, Yasuko Yoshida： Kobe Univ.

____

## Related Links
- [SHAPE](http://lbm.ab.a.u-tokyo.ac.jp/~iwata/shape/index.html)
- [Anacondaのインストール(Win,Mac,Linux)](https://www.python.jp/install/anaconda/index.html)

