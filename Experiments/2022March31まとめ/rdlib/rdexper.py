# rdexper.py 実験用
import pickle
from glob import glob
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from collections import deque
import optuna

# from sympy import *
from sympy import diff, Symbol, Matrix, symbols, solve, simplify, binomial, Abs, im, re, lambdify
from sympy.abc import a, b, c
# init_session()
from sympy import var

import tensorflow as tf
from keras import optimizers
from scipy.special import comb
import scipy.stats as stats

import datetime
import time
import random

# px,py =var('px:4'),var('py:4')

# OpenCV のファイル入出力が2バイト文字パス名に対応していないための対処
# （参考）https://qiita.com/SKYS/items/cbde3775e2143cad7455


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def assertglobal(params, verbose=False):
    global CONTOURS_APPROX, HARRIS_PARA, CONTOURS_APPROX, SHRINK, \
        HARRIS_PARA, GAUSSIAN_RATE1, GAUSSIAN_RATE2, UNIT, RPARA  # , WANDB, PATIENCE,PATIENCET,err_th
    for item in params:
        if item == 'UNIT':
            UNIT = params[item]  # 最終的に長い方の辺をこのサイズになるよう拡大縮小する
        elif item == 'SHRINK':
            SHRINK = params[item]  # 収縮膨張で形状を整える時のパラメータ
        elif item == 'CONTOURS_APPROX':
            CONTOURS_APPROX = params[item]  # 輪郭近似精度
        elif item == 'HARRIS_PARA':
            # ハリスコーナー検出で、コーナーとみなすコーナーらしさの指標  1.0 なら最大値のみ
            HARRIS_PARA = params[item]
        elif item == 'CONTOURS_APPROX':
            CONTOURS_APPROX = params[item]  # 輪郭近似精度
        elif item == 'GAUSSIAN_RATE1':
            GAUSSIAN_RATE1 = params[item]  # 先端位置を決める際に使うガウスぼかしの程度を決める係数
        elif item == 'GAUSSIAN_RATE2':
            GAUSSIAN_RATE2 = params[item]  # 仕上げに形状を整えるためのガウスぼかしの程度を決める係数
        elif item == 'RPARA':
            RPARA = params[item]  # 見込みでサーチ候補から外す割合
        # elif item == 'WANDB':
        #    WANDB = params[item]  # 見込みでサーチ候補から外す割合
        # elif item == 'PATIENCE':
        #    PATIENCE = params[item] # 曲線当てはめの際にこの回数誤差が増えたら処理を打ち切る
        # elif item == 'PATIENCET':
        #    PATIENCET = params[item] # tensorflow版曲線当てはめの際にこの回数誤差が増えたら処理を打ち切る
        # elif item == 'err_th':
        #    ERRORTOLERANCE = params[item] # 曲線当てはめで目標とする平均誤差
        # if verbose:
        #     print(item, "=", params[item])


assertglobal(params={
    'HARRIS_PARA': 1.0,  # ハリスコーナー検出で、コーナーとみなすコーナーらしさの指標  1.0 なら最大値のみ
    'CONTOURS_APPROX': 0.0002,  # 輪郭近似精度
    'SHRINK': 0.8,  # 0.75 # 収縮膨張で形状を整える時のパラメータ
    'GAUSSIAN_RATE1': 0.2,  # 先端位置を決める際に使うガウスぼかしの程度を決める係数
    'GAUSSIAN_RATE2': 0.1,  # 仕上げに形状を整えるためのガウスぼかしの程度を決める係数
    'UNIT': 256,  # 最終的に長い方の辺をこのサイズになるよう拡大縮小する
    'RPARA': 1.0,  # 見込みサーチのサーチ幅全体に対する割合 ３０なら左に３０％右に３０％の幅を初期探索範囲とする
    # 'WANDB':True # WANDB でログ
    # 'PATIENCE':10, # ベジエ曲線あてはめで、この回数続けて誤差が増えてしまったら探索を打ち切る
    # 'PATIENCET':10, # tensorflow版ベジエ曲線あてはめで、この回数続けて最小が更新されなかったら探索を打ち切る
    # 'ERRORTOLERANCE':0.75, # 曲線当てはめで目標とする平均誤差
})

# OpenCV バージョン３とバージョン４の輪郭抽出関数の違いを吸収する関数


def cv2findContours34(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE):
    v = list(cv2.__version__)
    if v[0] == '3':
        _img, cont, hier = cv2.findContours(
            image=image, mode=mode, method=method)
    else:  # if v[0] == '4':å
        cont, hier = cv2.findContours(
            image=image.copy(), mode=mode, method=method)
    return cont, hier

# (1)画像のリストアップ
# 指定フォルダ内の画像の収集
# 画像パスの収集


def collectimagepaths(path, imgexts=['jpg', 'jpge', 'png']):
    allfiles = glob(path+'/*')
    imgfiles = []
    for x in allfiles:
        if os.path.splitext(x)[-1][1:] in imgexts:
            imgfiles.append(x)
    imgfiles.sort()  # ファイル名の順に並び替える
    return imgfiles

# 画像の収集


def collectimages(path, imgexts=['jpg', 'jpge', 'png']):
    imgfiles = collectimagepaths(path, imgexts)
    imgs = [cv2.imread(afile, -1) for afile in imgfiles]
    return imgs

# サムネイルの作成


def makethumbnail(path, savedir='.', imgexts=['jpg', 'jpge', 'png']):
    imgfiles = collectimagepaths(path, imgexts)
    i = 0
    sam = Image.new('RGB', (500, 100*((len(imgfiles)+4)//5)), (0, 0, 0))
    row = col = 0
    for rad in imgfiles:
        img = Image.open(rad, 'r')
        img.thumbnail((100, 100))
        sam.paste(img, (col, row))
        col += 100
        if col == 500:
            col = 0
            row += 100
    dirname = os.path.splitext(os.path.basename(path))[0]
    sam.save(savedir+os.sep+'{}THUM.PNG'.format(dirname), 'PNG')
    return sam

# (2)画像の表示
# プロット用関数


def plotimg(img, layout=111):
    if img.ndim == 2:
        pltgry(img, layout)
    elif img.ndim == 3:
        pltcol(img, layout)


def pltgry(img, layout=111):
    plt.subplot(layout)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))


def pltcol(img, layout=111):
    plt.subplot(layout)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# (3) mkparaimage で２枚並べた画像を表示


def mkparaimage2(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img11 = np.zeros((h1, w1, 3), 'uint8')
    if img1.ndim == 2:
        img11[:, :, 0] = img11[:, :, 1] = img11[:, :, 2] = img1
    else:
        img11 = img1
    if img2.ndim == 2:
        img22 = np.zeros((h2, w2, 3), 'uint8')
        img22[:, :, 0] = img22[:, :, 1] = img22[:, :, 2] = img2
    else:
        img22 = img2
    paraimg = 255*np.ones((max(h1, h2), w1+w2+10, 3), dtype=np.uint8)

    paraimg[0:h1, 0:w1, :] = img11
    paraimg[0:h2, w1+10:, :] = img22

    return paraimg


def mkparaimage(imglist):
    if len(imglist) == 0:
        return
    if len(imglist) == 1:
        return imglist[0]
    if len(imglist) == 2:
        return mkparaimage2(imglist[0], imglist[1])
    return mkparaimage2(imglist[0], mkparaimage(imglist[1:]))


# (4) マージンをつける
MinimamMargin = 10


def makemargin(img, mr=1.5, mm=MinimamMargin):
    # 画像サイズが元の短径の mr 倍になるようにマージンをつける
    h, w = img.shape[:2]
    margin = int((mr-1)*min(h, w)//2 + 1)
    margin = mm if margin < mm else margin
    w2 = w + 2*margin
    h2 = h + 2*margin
    x1, y1 = margin, margin
    if len(img.shape) == 2:
        img2 = np.zeros((h2, w2), np.uint8)
    else:
        img2 = np.zeros((h2, w2, img.shape[2]), np.uint8)
    img2[y1:y1+h, x1:x1+w] = img
    return img2

# (4)-2 マージンのカット
# 白黒画像(0/255)が前提


def cutmargin(img, mr=1.0, mm=0, withRect=False):
    # default ではバウンディングボックスで切り出し
    # makemargin(切り出した画像,mr,mm)でマージンをつけた画像を返す
    # withRect = True の場合は切り出しの rect も返す
    if len(img.shape) > 2:
        gryimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gryimg = img.copy()
    bimg, data, ami = getMajorWhiteArea0(img=gryimg)
    x, y = data[ami][0:2]
    w, h = data[ami][2:4]
    bimg = bimg[y:y+h, x:x+w]  # マージンなしで切り出して返す
    bimg = makemargin(bimg, mr=mr, mm=mm)
    if withRect:
        return bimg, x, y, w, h
    else:
        return bimg

# (5) 最大白領域の取り出し


def getMajorWhiteArea0(img, order=1):
    # order 何番目に大きい領域を取り出したいか
    # dilation 取り出す白領域をどれだけ多めにするか
    # binary 返す画像を２値化するかどうか
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # カラーの場合はグレイ化する
    _ret, bwimg = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # ２値化
    # 背景が最大の領域であるという前提を用いているが、そうでない可能性があるのでいったん黒を付け足す
    limg = np.zeros_like(bwimg)
    bwimg = np.c_[bwimg, limg]
    _lnum, labelimg, cnt, _cog = cv2.connectedComponentsWithStats(
        bwimg)  # ラベリング
    areaindexs = np.argsort(-cnt[:, 4])
    if len(areaindexs) > order:
        areaindex = areaindexs[order]  # order 番目に大きい白領域のインデックス
    else:
        areaindex = areaindexs[-1]  # 指定した番号のインデックスが存在しないなら一番小さい領域番号
    labelimg[labelimg != areaindex] = 0
    labelimg[labelimg == areaindex] = 255
    w = labelimg.shape[1]//2
    labelimg = labelimg.astype(np.uint8)[:, :w]
    return labelimg, cnt, areaindex

# (5)-2


def getMajorWhiteArea(img, order=1, dilation=0, binary=False):
    # order 何番目に大きい領域を取り出したいか
    # dilation 取り出す白領域をどれだけ多めにするか
    # binary 返す画像を２値化するかどうか
    limg, _cnt, _index = getMajorWhiteArea0(img=img, order=order)

    if binary:
        oimg = limg.copy()
    else:
        oimg = img.copy()
    if dilation > 0:
        k = calcksize(limg)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fatlabelimg = cv2.dilate(limg, kernel, iterations=dilation)  # 太らせる
        oimg[fatlabelimg == 0] = 0
    else:
        oimg[limg == 0] = 0
    if binary:
        oimg[limg > 0] = 255
    return oimg

# (6) 処理結果画像（fimg)に処理前画像（bimg)の輪郭を描く


def draw2(bimg, fimg, thickness=2, color=(255, 0, 200)):
    bimg2 = getMajorWhiteArea(bimg, binary=True)
    if len(fimg.shape) > 2:
        fimg2 = fimg.copy()
    else:
        fimg2 = cv2.cvtColor(fimg, cv2.COLOR_GRAY2BGR)
    # 処理前画像の輪郭
    # canvas = np.zeros_like(fimg2)
    canvas = fimg2.copy()
    _ret, bwimg = cv2.threshold(
        bimg2, 128, 255, cv2.THRESH_BINARY)  # 白画素は255にする
    cnt, _hierarchy = cv2findContours34(
        bwimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = cv2.drawContours(canvas, cnt, -1, color, thickness)
    return canvas

# (7) (x1,y1)から（x2,y2) に向かう直線のX軸に対する角度


def getDegreeOfALine(x1, y1, x2, y2):
    dx = x2-x1
    dy = y2-y1
    if dx == 0:
        if dy > 0:
            deg = 90
        else:
            deg = -90
    elif dx > 0:
        deg = 180.0*np.arctan(dy/dx)/np.pi
    else:
        deg = 180*(1+np.arctan(dy/dx)/np.pi)
    return deg

# (8) (x1,y1)から（x2,y2) に向かう直線の延長上の十分離れた２点の座標


def getTerminalPsOnLine(x1, y1, x2, y2):
    dx = x2-x1
    dy = y2-y1
    s1 = int(x1 + 1000*dx)
    t1 = int(y1 + 1000*dy)
    s2 = int(x1 - 1000*dx)
    t2 = int(y1 - 1000*dy)
    return s1, t1, s2, t2

# (9) ガウスぼかし、膨張収縮、輪郭近似で形状を整える関数 RDreform()
# 形状の細かな変化をガウスぼかし等でなくして大まかな形状にする関数

# (9)-1 ガウスぼかしの程度を決めるカーネルサイズの自動決定


def calcksize(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(img, (7, 7), 0)  # とりあえず (7,7)でぼかして2値化
    _ret, bwimg = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # ２値化
    _lnum, labelimg, cnt, _cog = cv2.connectedComponentsWithStats(
        bwimg)  # ラベリング
    if(len(cnt) < 2):
        ksize = 3
    else:
        areamax = np.argmax(cnt[1:, 4])+1  # ０番を除く面積最大値のインデックス
        maxarea = np.max(cnt[1:, 4])
        ksize = int(np.sqrt(maxarea)/60)*2+1
    return ksize

# (9)-2


def RDreform(img, order=1, ksize=0, shrink=SHRINK):
    # ksize : ガウスぼかしの量、shrink 膨張収縮による平滑化のパラメータ
    # order : 取り出したい白領域の順位
    # shrink : 膨張収縮による平滑化のパラメータ　

    # ガウスぼかしを適用してシルエットを滑らかにする
    # ガウスぼかしのカーネルサイズの決定

    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.sum() == 0:  # 白領域が存在しない
        return img
    if ksize == 0:  # ぼかしのサイズが指定されていないときは最大白領域の面積を基準に定める
        ksize = calcksize(img)
    img2 = cv2.GaussianBlur(img, (ksize, ksize), 0)  # ガウスぼかしを適用
    img2 = getMajorWhiteArea(
        img2, order=order, dilation=2)  # 指定白領域を少しだけ大きめに取り出す

    _ret, img2 = cv2.threshold(
        img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # ２値化

    return RDreform_D(img2, ksize=ksize, shrink=SHRINK)

# (9)-3


def RDreform_D(img, ksize=5, shrink=SHRINK):

    # 収縮・膨張によりヒゲ根を除去する
    area0 = np.sum(img)  # img2 の画素数*255 になるはず
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ksize, ksize))  # 円形カーネル

    # ミディアンフィルタで小さなノイズを除去
    img = cv2.medianBlur(img, ksize)

    # 残った穴を埋める
    h, w = img.shape[:2]
    inv = 255-img
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    _retval, holes, _mask, _rect = cv2.floodFill(inv, mask, (0, 0), newVal=0)
    img = img + holes

    tmpimg = cv2.erode(img, kernel, iterations=1)  # 収縮１回目
    area1 = np.sum(tmpimg)  # 収縮したので area0 より少なくなる

    n = 1  # 収縮回数のカウンタ

    while area0 > area1 and area1 > shrink*area0:  # 面積が SHRINK倍以下になるまで繰り返す
        tmpimg = cv2.erode(tmpimg, kernel, iterations=1)
        area1 = np.sum(tmpimg)
        n += 1
    img3 = cv2.dilate(tmpimg, kernel, iterations=n)  # 同じ回数膨張させる
    # あらためて輪郭を求め直す

    cnt, _hierarchy = cv2findContours34(
        img3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # あらためて輪郭を抽出
    outimg = np.zeros_like(img3)
    if len(cnt) > 0:
        perimeter = cv2.arcLength(cnt[0], True)  # 周囲長
        epsilon = CONTOURS_APPROX*perimeter  # 周囲長をもとに精度パラメータを決定
        # 概形抽出
        approx = cv2.approxPolyDP(cnt[0], epsilon, True)

        # 輪郭を描いて埋める
        outimg = cv2.drawContours(outimg, [approx], 0, 255, thickness=-1)
    else:
        outimg = np.ones_like(img3)*255
    return outimg

# (10) Grabcut による大根領域の抜き出し
# GrabCutのためのマスクを生成する


def mkGCmask(img, order=1):
    # カラー画像の場合はまずグレー画像に変換
    gray = RDreform(img, order=order, ksize=0, shrink=SHRINK)

    # 大きめのガウシアンフィルタでぼかした後に大津の方法で２階調化
    ksize = calcksize(gray)  # RDForm で使う平滑化のカーネルサイズ
    bsize = ksize
    blur = cv2.GaussianBlur(gray, (bsize, bsize), 0)  # ガウスぼかし
    coreimg = getMajorWhiteArea(blur)  # ２値化して一番大きな白領域
    coreimg[coreimg != 0] = 255

    # 膨張処理で確実に背景である領域をマスク
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ksize, ksize))  # 円形カーネル
    mask1 = 255-cv2.dilate(coreimg, kernel, iterations=ksize)

    # 収縮処理で確実に内部である領域をマスク
    mask2 = cv2.erode(coreimg, kernel, iterations=ksize)

    return mask1, mask2

# (11) 大根部分だけセグメンテーションし、結果とマスクを返す


def getRadish(img, order=1, shrink=SHRINK):
    # 白領域の面積が order で指定した順位の領域を抜き出す

    mask1, mask2 = mkGCmask(img, order=order)

    # grabcut　用のマスクを用意
    grabmask = np.ones(img.shape[:2], np.uint8)*2
    #
    grabmask[mask1 == 255] = 0     # 黒　　背景　　
    grabmask[mask2 == 255] = 1    # 白　前景

    # grabcut の作業用エリアの確保
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # MASK による grabcut 実行
    grabmask, bgdModel, fgdModel = cv2.grabCut(
        img, grabmask, None, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_MASK)
    grabmask = np.where((grabmask == 2) | (
        grabmask == 0), 0, 1).astype('uint8')
    grabimg = img*grabmask[:, :, np.newaxis]
    silimg = np.zeros(grabimg.shape[:2], np.uint8)
    graygrabimg = cv2.cvtColor(grabimg, cv2.COLOR_BGR2GRAY)
    silimg[graygrabimg != 0] = 255

    silimg = getMajorWhiteArea(silimg, order=1, dilation=0)
    silimg = RDreform_D(silimg, ksize=calcksize(silimg), shrink=shrink)

    return grabimg, silimg

# (12) 重心の位置を求める


def getCoG(img):
    _lnum, _img, cnt, cog = cv2.connectedComponentsWithStats(img)
    areamax = np.argmax(cnt[1:, 4])+1  # ０番を除く面積最大値のインデックス
    c_x, c_y = np.round(cog[areamax])  # 重心の位置を丸めて答える
    # x,y,w,h,areas = cnt[areamax] # 囲む矩形の x0,y0,w,h,面積
    return c_x, c_y, cnt[areamax]

# (13) 回転した上でマージンをカットした画像を返す


def rotateAndCutMargin(img, deg, c_x, c_y):
    # 非常に稀であるが、回転すると全体が描画領域外に出ることがあるので作業領域を広く確保
    # mat = cv2.getRotationMatrix2D((x0,y0), deg-90, 1.0) # アフィン変換マトリクス
    bigimg = makemargin(img, mr=10)  # 作業用のマージンを確保
    h3, w3 = img.shape[:2]
    h4, w4 = bigimg.shape[:2]

    if deg != 0:
        mat = cv2.getRotationMatrix2D(
            (c_x+(w4-w3)/2, c_y+(h4-h3)/2), deg, 1.0)  # アフィン変換マトリクス
        # アフィン変換の適用
        bigimg = cv2.warpAffine(bigimg, mat, (0, 0), 1)

    _ret, bigimg = cv2.threshold(bigimg, 127, 255, cv2.THRESH_BINARY)  # 再２値化

    # 再び最小矩形を求めて切り出す。
    _nLabels, _labelImages, data, _center = cv2.connectedComponentsWithStats(
        bigimg)
    ami = np.argmax(data[1:, 4])+1  # もっとも面積の大きい連結成分のラベル番号　（１のはずだが念の為）
    resultimg = bigimg[data[ami][1]:data[ami][1]+data[ami]
                       [3], data[ami][0]:data[ami][0]+data[ami][2]]

    return resultimg

# (14) 大きさを正規化したシルエットの生成


def getNormSil(img, tiltzero=True, mr=1.5, unitSize=UNIT):
    #  img 入力画像
    #  tiltzero  True なら傾き補正する
    #  mr マージンなし画像に対するマージンあり画像のサイズ比率
    #  unitSize バウンダリの長辺をこの値にする

    img = img.copy()

    if tiltzero:  # 傾き補正する
        img = tiltZeroImg(img)

    return makeUnitImage(img, mr=mr, unitSize=UNIT)

# (15) 長辺の mr 倍サイズの枠の中央に対象を配置した画像を返す


def makeUnitImage(img, mr=1.5, unitSize=UNIT):
    # 長辺が UNIT ピクセルになるよう縮小し、(mrxUNIT)x(mrxUNIT)の画像の中央に配置する。
    h, w = img.shape[:2]
    if w > h:
        rsh = int(round(h * UNIT/w))
        rsw = UNIT
    else:
        rsh = UNIT
        rsw = int(round(w * UNIT/h))
    rimg = cv2.resize(img, (rsw, rsh))  # リサイズの実行

    MSIZE = int((UNIT*(mr-1))//2)+1  # マージン
    IMGSIZE = UNIT+2*MSIZE  # 画像サイズ
    x0 = int((IMGSIZE-rsw)//2)  # はめ込みの基準点
    y0 = int((IMGSIZE-rsh)//2)
    canvas = np.zeros((IMGSIZE, IMGSIZE), np.uint8)  # キャンバスの確保
    canvas[y0:y0+rsh, x0:x0+rsw] = rimg  # リサイズして中央にはめ込み
    _ret, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
    return canvas

# (16) 近似楕円の軸方向が水平垂直となるように回転補正した画像を求める


def tiltZeroImg(img):
    h, w = img.shape[:2]
    img0, rx, ry, rw, rh = cutmargin(
        img, mm=0, withRect=True)  # バウンダリボックスで切り出し
    cnt = cv2findContours34(img0, 1, 2)[0][0]  # 輪郭線抽出
    (gx0, gy0), (sr, lr), ang = cv2.fitEllipse(cnt)  # 楕円近似
    #  fitEllipse の返す角度は長軸を垂直に立てるための角度である。
    #  もしも幅が高さより大きい個体であれば、回転角を９０度減ずる
    ang = ang-90 if rw > rh else ang

    # 重心の偏りぐあいを記録
    ghight0 = abs(rh/2 - gy0)  # シルエット重心と矩形中心のy方向距離
    gright0 = abs(rw/2 - gx0)  # シルエット重心と矩形中心のx方向距離

    # 近似楕円の軸が水平垂直となるよう重心周りに回転を加えた図形を求める
    img1 = rotateAndCutMargin(img0, ang, gx0, gy0)

    # あらためて重心位置を求める
    gx1, gy1, (_, _, w1, h1, _) = getCoG(img1)

    # 180度回転してしまう場合があるので、重心の偏りぐあいで判断し、必要なら回転
    if ghight0 > gright0:  # 重心の上下の偏りがと左右の偏りより大きい場合
        if (rh/2 > gy0) != (h1/2 > gy1):  # 上下の
            img1 = np.rot90(np.rot90(img1))
    else:
        if (rw/2 > gx0) != (w1/2 > gx1):
            img1 = np.rot90(np.rot90(img1))

    return img1

# (17) 輪郭点列を得る


def getContour(img):
    # 輪郭情報 主白連結成分の輪郭点列のみ返す関数
    contours, _hierarchy = cv2findContours34(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 輪郭線追跡
    cnt00 = contours[np.argmax([len(c) for c in contours])]  # 最も長い輪郭
    return cnt00

# (18) 幅１の輪郭データを開く


def openAContour(cnt):
    # cnt 幅１の領域の輪郭データ
    cnt0 = cnt.squeeze()  # すでに squeeze()されていた場合は変化しない
    # 分岐のない線図形の輪郭は、トレースが端点から始まれば１箇所、途中からなら２箇所折り返しがある。端点と折り返し、
    # もしくは、折り返しと折り返しの間を取り出すことで、重複のない輪郭データとする
    i1 = 0
    for i in range(int(len(cnt0))-1):
        if np.all(cnt0[i-1] == cnt0[i+1]) or np.all(cnt0[i-2] == cnt0[i+1]):
            i0, i1 = i1, i
    cnt0 = cnt0[i0:i1+1]
    if cnt0[0][1] > cnt0[-1][1]:
        cnt0 = cnt0[::-1]

    return cnt0

# (19) 輪郭表現の相互変換
# 輪郭構造体をただのリストに変換  -> [[1,2],[2,3]....]


def contolist(con):
    return con.squeeze().tolist()

# リストを輪郭線構造体に変換  -> array([[[1,2]],[[2,3]],...])


def listtocon(list):
    return np.array([[p] for p in list])

# (20)  輪郭の描画


def drawContours(canvas, con, color=255, thickness=1):
    if type(con) == np.ndarray:
        if con.ndim == 3:  # 普通の輪郭情報
            cv2.drawContours(canvas, con, -1, color=color, thickness=thickness)
        elif con.ndim == 2:  # 片側のみの輪郭
            cv2.polylines(canvas, [con], isClosed=False,
                          color=color, thickness=thickness)
    elif type(con) == list:
        for c in con:
            drawContours(canvas, c, color=255, thickness=1)

# (21) 曲率関数


def curvature(func):  # func は sympy 形式の t の関数（fx,fy）のペア
    t = symbols('t')
    fx, fy = func
    dx = diff(fx, t)
    dy = diff(fy, t)
    ddx = diff(dx, t)
    ddy = diff(dy, t)
    k = (dx*ddy - dy*ddx)/(dx*dx + dy*dy)**(3/2)
    return -k

# 画像の座標系は数学の座標系とｙ方向が逆なので正負が反転する
# (22) 輪郭中の曲率最大点のインデックスと輪郭データを返す


def maxCurvatureP(rdimg, con=[], cuttop=0, cutbottom=0.8, sband=0.25, N=8):
    # rdimg 画像、con 輪郭データ
    # cuttop, cutbottom 個体の高さに対してこの範囲は除外する
    # sband 曲率最大値を探す範囲　0.5 を挟んで両側この値の範囲で曲率最大点を探す
    con = con.squeeze()
    # 輪郭データが与えられていないなら抽出
    if len(con) == 0:
        con = getContour(rdimg)
    ys = con[:, 1]
    y0 = ys.min()  # バウンディングボックスの上端
    h = ys.max() - y0 + 1  # バウンディングボックスの高さ
    canvas = np.zeros_like(rdimg)
    drawContours(canvas, con)  # 輪郭を描く
    canvas[int(y0 + cuttop*h):int(y0 + cutbottom*h), :] = 0  # 指定範囲を黒で塗りつぶす
    # 輪郭の輪郭を抽出

    cnt0 = openAContour(getContour(canvas))
    bez = BezierCurve(N=N, samples=cnt0)  # 7次近似固定
    cps, [fx, fy] = bez.fit0()  # ベジエ近似
    t = symbols('t')
    kf = curvature([fx, fy])  # 曲率関数を得る
    # t = 0.5 ± sband 内 を１００等分して曲率データを得る
    ps = np.array([float(kf.subs(t, s))
                   for s in np.linspace(0.5-sband, 0.5+sband, 101)])
    maxindex = ps.argmax()  # 曲率最大のデータ番号
    tmax = 0.5 - sband + 2*sband*maxindex/100
    mx, my = float(fx.subs(t, tmax)), float(fy.subs(t, tmax))  # 曲率最大点の座標
    maxindex = np.array([np.linalg.norm(v) for v in con - [mx, my]]).argmin()
    return maxindex, con

# (23) 中心軸端点の推定


def findTips(img, con=[], top=0.1, bottom=0.8, topCD=0.5, bottomCD=0.5, mode=2):
    # 入力　
    #   img シルエット画像
    #   con 輪郭点列　（なければ画像から作る）
    # パラメータ
    #   top 中心軸上端点の候補探索範囲　高さに対する割合
    #   bottom 中心軸下端点の候補探索範囲　高さに対する割合
    #   topCD  中心軸上端点らしさの評価データを収集する範囲
    #   bottomCD  中心軸下端点らしさの評価データを収集する範囲
    #   mode 0: 頭頂点のみ返す、1:尾端のみ返す、2:頭頂と尾端の両方を返す
    # 出力
    #   con 輪郭点列
    #   topTip  中心軸上端点の輪郭番号
    #   bottomTip 中心軸下端点の輪郭番号
    #   symtops,symbottoms 評価データ

    if len(con) == 0:  # 点列がすでにあるなら時間短縮のために与えてもよい
        con = getContour(img)
    conlist = con.squeeze().tolist()  # 輪郭点列のリスト
    gx, gy, (x0, y0, w, h, a) = getCoG(img)  # 重心とバウンディングボックス
    ncon = len(conlist)  # 輪郭点列の数)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # オープニング（収縮→膨張）平滑化した図形を求める
    img1 = cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, kernel, iterations=5)
    HL = (ncon+len(getContour(img1)))//4  # 平滑化図形の輪郭長

    # 対称性評価関数
    import cmath

    def calcval(i, irrend, isBottom=False):
        data = []
        vc = 0
        for n in range(3, irrend):  # 距離が近すぎると誤差が大きいので３から

            p0, p1, p2 = conlist[i], conlist[(
                i-n) % ncon], conlist[(i+n) % ncon]
            v1 = (p2[0]-p1[0], p2[1]-p1[1])  # p1p2 ベクトル
            p3 = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)  # p3 = p1とp2の中点
            v2 = (p3[0]-p0[0], p3[1]-p0[1])  # p0p3 ベクトル
            # v1 の複素表現  画像データは下がプラスなので虚成分を反転して考えないといけない
            pv1 = v1[0] - 1j*v1[1]
            pv2 = v2[0] - 1j*v2[1]  # v2 の複素表現
            nv1 = abs(pv1)  # v1 の長さ
            nv2 = abs(pv2)  # v2 の長さ
            if nv2 > 3 and nv1 > 3:  # 距離が近すぎると誤差が大きいので３以上
                if isBottom:  # そこの場合は凹部は選ばない
                    ph1 = cmath.phase(pv1)  # v1 の偏角
                    ph2 = cmath.phase(pv2)  # v2 の偏角
                    ang2 = ph2 - ph1 if ph2 > ph1 else ph2 - ph1 + 2*cmath.pi  # v1 と v2 のなす角（ラジアン）
                    if ang2 < cmath.pi:  # 凸の場合　cos の値をペナルティとする＝直角ならペナルティ０
                        data.append(abs(np.dot(v1, v2)/nv1/nv2))
                    else:  # 凹の場合はペナルティ1
                        data.append(1.0)
                else:  # Top
                    data.append(abs(np.dot(v1, v2)/nv1/nv2))
        if len(data) == 0:
            return -1
        else:
            return mean(data)

    # 上部の端点の探索
    symtops = []
    if mode == 1:
        topTip = 0
    else:
        for i in range(ncon):
            if conlist[i][1] - y0 >= top*h:  # バウンディングボックス上端からの距離
                val = -1
            else:
                val = calcval(i, irrend=int(topCD*HL), isBottom=False)
            symtops.append(val)
        m = np.max(symtops)  # 探索対象のうちの最大評価値
        for i in range(ncon):
            if symtops[i] < 0:
                symtops[i] = m
        topTip = np.argmin(symtops)

    # 下部の端点は曲率最大点
    # bottomTip, _con = maxCurvatureP(img,con=con,cutbottom = bottom, sband=sband,N=N)

    # 下部の端点の探索 （上端と同じ手法）

    symbottoms = []
    if mode == 0:
        bottomTip = 0
    else:
        for i in range(ncon):
            if conlist[i][1] - y0 < bottom*h:  # バウンディングボックス上端からの距離
                val = -1
            else:
                val = calcval(i, irrend=int(bottomCD*HL), isBottom=True)
            symbottoms.append(val)
        m = np.max(symbottoms)  # 探索対象のうちの最大評価値
        for i in range(ncon):
            if symbottoms[i] < 0:
                symbottoms[i] = m
        bottomTip = np.argmin(symbottoms)

    return con, topTip, bottomTip, symtops, symbottoms

# (24) 上端・末端情報に基づき輪郭線を左右に分割する


def getCntPairWithCntImg(rdcimg, dtopx, dtopy, dbtmx, dbtmy, dtopdr=3, dbtmdr=3, mode=2):
    # drcimg: ダイコンの輪郭画像
    # (dtopx,dtopy) dtopdr　上部削除円中心と半径
    # (dbtmx,dbtmy) dbtmdr 　下部削除円中心と半径
    # mode:  1:下部端点で開いた輪郭を返す、2: 左右分割した2本の輪郭を返す
    # if  dtopdr < 2 or dbtmdr < 2:
    #    print("Warning. Radius may be too small. dtopdr {}, dbtmdr {}".format(dtopdr,dbtmdr))
    # 中心軸上端部と末端部に黒で円を描いて輪郭を２つに分離
    canvas = rdcimg.copy()
    # まず上端を指定サイズの円で削る
    if mode != 1:  # 下端のみが要求されているときはパス
        canvas = cv2.circle(canvas, (dtopx, dtopy), dtopdr, 0, -1)

    if mode != 0:  # 上端のみが要求されているときはパス
        while True:
            # 次に末端を削る。末端は細いので、左右の輪郭が縮退している場合があり、削除円が小さいと輪郭が分離できず処理が進められない。
            canvas = cv2.circle(canvas, (dbtmx, dbtmy), dbtmdr, 0, -1)

            # 輪郭検出すれば２つの輪郭が見つかるはず。
            nLabels, _labelImages = cv2.connectedComponents(canvas)
            if mode == 2 and nLabels >= 3:  # 背景領域を含めると３以上の領域になっていれば正しい
                break
            elif mode != 2 and nLabels >= 2:
                break
            dbtmdr = dbtmdr + 2  # ラベル数が　３（背景を含むので３） にならないとすれば先端が削り足りない可能性が最も高いので半径を増やしてリトライ

    contours, hierarchy = cv2findContours34(
        canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxcnt_i = np.argmax(np.array([len(c) for c in contours]))
    cnt0 = contours[maxcnt_i].squeeze()  # 最も長い輪郭

    if mode == 2:
        contours = contours[:maxcnt_i]+contours[maxcnt_i+1:]
        maxcnt_i = np.argmax(np.array([len(c) for c in contours]))
        cnt1 = contours[maxcnt_i].squeeze()

    # 分岐のない線図形の輪郭は、トレースが端点から始まれば１箇所、途中からなら２箇所折り返しがある。端点と折り返し、
    # もしくは、折り返しと折り返しの間を取り出すことで、重複のない輪郭データとする
    cnt0 = openAContour(cnt0)

    if mode != 2:
        return cnt0
    else:  # mode == 2:
        cnt1 = openAContour(cnt1)

        # 中程の点を比べて左にある方を左と判定する。
        c0 = cnt0[int(len(cnt0)/2)][0]
        c1 = cnt1[int(len(cnt1)/2)][0]
        if c0 > c1:
            conLeft, conRight = cnt1, cnt0
        else:
            conRight, conLeft = cnt1, cnt0
        return conLeft, conRight

# (25) 与えられたダイコン画像の輪郭を左右に分割する


def getCntPairWithImg(rdimg, top=0.1, bottom=0.9, topCD=0.9, bottomCD=0.2, dtopdr=3, dbtmdr=3, mode=2):
    # drimg: ダイコンの画像
    # top,bottom,topCD：findTips() に与えるパラメータ
    # dtopdr,dbtmdr:　getCntPairWithCntImg() に与えるパラメータ
    con, topTip, bottomTip, symtops, symbottoms = findTips(
        rdimg, top=top, bottom=bottom, topCD=topCD, bottomCD=bottomCD, mode=mode)
    conlist = con.squeeze()
    dtopx, dtopy = conlist[topTip]
    dbtmx, dbtmy = conlist[bottomTip]
    rdcimg = np.zeros_like(rdimg)  # 描画キャンバスの準備
    cv2.drawContours(rdcimg, con, -1, 255, thickness=1)
    if mode == 2:
        conLeft, conRight = getCntPairWithCntImg(
            rdcimg, dtopx, dtopy, dbtmx, dbtmy, dtopdr=dtopdr, dbtmdr=dbtmdr, mode=mode)
        return conLeft, conRight
    else:
        contAll = getCntPairWithCntImg(
            rdcimg, dtopx, dtopy, dbtmx, dbtmy, dtopdr=dtopdr, dbtmdr=dbtmdr, mode=mode)
        return contAll

# (26) 座標リストから等間隔で指定した数の標本を抜き出す。
def getSamples(cont, N=20, mode='Equidistant'):
    if mode == 'Equidistant':
        axlength = np.array(cv2.arcLength(cont, closed=False))  # 弧長
        lengths = np.array([cv2.arcLength(cont[:i+1], closed=False)
                            for i in range(len(cont))])  # 各点までの弧長の配列
        return np.array([cont[np.abs(lengths - i).argmin()] for i in np.linspace(0, axlength, N)])
    else:  # 'Simple' 実際にはなんでもOK
        return cont[list(map(int, np.linspace(0, len(cont)-1, N)))]

#  (27) N次ベジエフィッティング
#from sympy import diff,Symbol,Matrix,symbols,solve,simplify,binomial
#from sympy.abc import a,b,c
#from sympy import var

# 輪郭点列中の両端の２標本点間の点列の序列中央の座標を返す関数
def getMutPoints(cont=[],Samples=[]):
    if len(Samples)>0:
        wx1 = np.where(cont[:,0]==Samples[1][0])[0] # X座標が１番と一致する要素番号
        wy1 = np.where(cont[:,1]==Samples[1][1])[0] # Y座標が１番と一致する要素番号
        im1 = wx1[np.in1d(wx1, wy1)][0] # Sample[1]のcontにおけるインデックス
        wx2 = np.where(cont[:,0]==Samples[-2][0])[0] # X座標が-2番と一致する要素番号
        wy2 = np.where(cont[:,1]==Samples[-2][1])[0] # Y座標が-2番と一致する要素番号
        im2 = wx2[np.in1d(wx2, wy2)][0] # Sample[1]のcontにおけるインデックス
        return [cont[im1//2],cont[(im2+len(cont)-1)//2]]
    else:
        return []

#  稠密なパラメータを得る（点列は一定間隔にならないので、点列が一定間隔になるようなパラメータ列を求める）
def getDenseParameters(func, st=0.0, et=1.0, n_samples=31, span=0, needlength=False):
    # func 曲線のパラメータ表現 = (fx,fy)
    # n_samples 必要なパラメータ数 = サンプル数
    # span 稠密さの係数　1 なら 候補パラメータの刻みが１画素以内に収まるように　２なら２画素以内に…
    if func == None:  # 近似式が与えられていない場合
        return np.linspace(0, 1, n_samples)
    else:
        fx,fy = func
        dfx, dfy = diff(fx), diff(fy)
        nfx, nfy = lambdify('t', fx, "numpy"), lambdify('t', fy, "numpy")
        ndfx, ndfy = lambdify('t', dfx, "numpy"), lambdify('t', dfy, "numpy")
        para = [st]  # 候補パラメータを格納するリスト　
        bzpoints = [[int(nfx(st)), int(nfy(st))]]  # パラメータに対応する座標のリスト
        if span == 0:  # 稠密さの係数が与えられていない場合はサンプル30点でおおざっぱに全長を見積もって決める
            ss = np.linspace(st, et, 30)
            ps = np.array([[int(nfx(s)), int(nfy(s))] for s in ss])
            axlength = cv2.arcLength(ps, closed=False) # 経路長
            # 経路長が実際の長さより短めの値に算出される。その３分の１なので実際のサンプル間の長さの３分の１以下になる
            span = axlength/(n_samples-1)/3
        s1 = st
        while s1 < et:
            absdx = abs(ndfx(s1))  # x微係数
            absdy = abs(ndfy(s1))  # y微係数
            absd = np.sqrt(float(absdx**2 + absdy**2))  # 傾き
            pstep = span/absd if absd > 0 else 1/n_samples  # 傾きの逆数＝ｘかｙが最大span移動するだけのパラメータ変化
            s1 += 0.7*pstep  # span を超えないよう、７掛けで控えめにパラメータを増やす　
            s1 = et if s1 > et else s1
            para.append(s1)  # リストへ追加
            # s1 に対応する曲線上の点をリストに追加
            bzpoints.append([int(nfx(s1)), int(nfy(s1))])
        bzpoints = np.array(bzpoints)
        axlength = cv2.arcLength(bzpoints, closed=False)  # 弧長
        lengths = np.array([0]+[cv2.arcLength(bzpoints[:i+1], closed=False)
                                for i in range(1, len(bzpoints))])  # 各点までの弧長の配
        tobefound = np.linspace(0, axlength, n_samples)  # 全長をサンプルの数で区切る
        ftpara = [st]
        found = [0.0]
        i = 1
        for slength in tobefound[1:-1]:
            while lengths[i] < slength:
                i += 1
            ftpara.append(float(para[i]))
            found.append(lengths[i])
        ftpara.append(et)
        found.append(axlength)
        if needlength:
            return ftpara, found
        else:
            return ftpara
        
# ベジエ曲線のクラス定義
class BezierCurve:
    # インスタンス変数
    # f [X座標関数式,Y座標関数式]
    # samples 標本点のリスト
    # ts 標本点に対するベジエパラメータ

    # クラス変数
    # xx driftThres = 0.01 # 繰り返しにおけるパラメータ変動幅の平均値に対するしきい値
    # xx errorThres = 0.01 # 繰り返しを打ち切る誤差変化量
    dCount = 3  # ２分探索の打ち切り回数 （3以上が望ましい）
    convg_coe = 1e-5 # 5.0e-8  # 収束の見切り　　１回あたりの誤差減少が少なすぎる時の打ち切り条件を決める値 3e-8〜8e-8 が最適
    mloop_itt = 3 # fit1T mode 0 の繰り返しループにおける、minimize() の繰り返し回数。
    debugmode = False
    AsymptoticPriority = 'distance'  # パラメータ更新法
    wandb = False
    openmode = False

    # 'distance':距離優先、'span':間隔優先

    def __init__(self, N=5, samples=[], prefunc=None, tpara=[]):
        self.samples = samples  # 標本点
        self.prefunc = prefunc  # 関数式
        # self.f  ベジエ曲線の式 = [fx,fy]
        # 'px0','px1'... が制御点のX座標を表すシンボル
        # 'py0','py1'...が制御点のY座標を表すシンボル
        self.N = N
        # ベジエ曲線を定義するのに使うシンボルの宣言
        P = [Symbol('P' + str(i)) for i in range(N+1)]  # 制御点を表すシンボル（数値変数ではない）
        px = [var('px'+str(i)) for i in range(N+1)]  # 制御点のx座標を表すシンボル
        py = [var('py'+str(i)) for i in range(N+1)]  # 制御点のy座標を表すシンボル
        t = symbols('t')
        v = var('v')
        for i in range(N+1):     # 制御点のシンボルと成分の対応付け
            P[i] = Matrix([px[i], py[i]])
        # N次のベジエ曲線の定義式制御点 P0～PN とパラメータtの関数として定義
        v = 1-t
        bezf = Matrix([0, 0])
        for i in range(0, N+1):
            bezf = bezf + binomial(N, i)*v**(N-i)*t**i*P[i]
        self.f = bezf
        # もし、inf データが含まれるならば、補間する（計算で求められた座標データがサンプルの場合にありうる）
        if len(samples) > 0:
            self.samples = self.interporation(samples)
        # 初期パラメータのセット
        if len(tpara) > 0:
            self.ts = tpara.copy()
        else:
            self.ts = self.assignPara2Samples(prefunc=prefunc)

    # 当てはめの自乗誤差の平均値を算出する関数
    def f_meanerr(self, fx, fy, ts):
        sps = self.samples
        # fx, fy : t の symfy関数、ts: ベジエのパラメータのリスト, sps サンプル点のリスト
        t = symbols('t')
        nfx, nfy = lambdify(t, fx, "numpy"), lambdify(t, fy, "numpy")
        onps = [[nfx(ts[i]), nfy(ts[i])] for i in range(len(ts))]
        return mean([(sps[i][0]-onps[i][0])**2+(sps[i][1]-onps[i][1])**2 for i in range(len(sps))])

    # 解なしの部分に np.inf が入っているのでその抜けを前後から推定してデータを埋める
    def interporation(self, plist):
        # plist : np.inf が混入している可能性のある座標の numpy array
        foundinf = 0
        while np.sum(plist) == np.inf:  # np.inf を含むなら除去を繰り返す
            for i in range(len(plist)):
                if np.sum(plist[i]) == np.inf:
                    foundinf += 1
                    print("欠", end="")
                    # 当該は無限で、前後は無限ではない場合
                    if (i != 0 and i != len(plist)-1) and np.sum(plist[i-1]+plist[i+1]) != np.inf:
                        plist = np.r_[plist[0:i], [
                            (plist[i-1]+plist[i+1])/2], plist[i+1:]]
                    elif len(plist[i:]) >= 3 and np.sum(plist[i+1]+plist[i+2]) != np.inf:
                        plist = np.r_[plist[0:i], [plist[i+2] -
                                                   2*(plist[i+2]-plist[i+1])], plist[i+1:]]
                    elif len(plist[0:i]) >= 2 and np.sum(plist[i-1]+plist[i-2]) != np.inf:
                        plist = np.r_[plist[0:i], [plist[i-2] -
                                                   2*(plist[i-2]-plist[i-1])], plist[i+1:]]
        if foundinf:
          print("A total of {} data are missing.".format(foundinf))
        return plist

    # サンプル点が等間隔になるようにパラメータを設定する
    def assignPara2Samples(self, prefunc=None, scatter=False):
        samples = self.samples 
        if len(samples) == 0:  # 標本点を与えずにベジエ曲線の一般式として使うこともできる
            return
        if prefunc != None:  # 近似式が与えられている場合
            return getDenseParameters(func=prefunc, n_samples=len(samples), span=0)
        # 曲線の式が不明である場合は、単純に０〜１を等間隔に刻んだ値を返す
        else:  # パラメータが与えられていない場合、0～1をリニアに各サンプル点までので経路長で刻む
            # axlength = np.array(cv2.arcLength(samples, False)) # 点列に沿って測った総経路長
            # 各サンプル点の始点からの経路長の全長に対する比を、各点のベジエパラメータの初期化とする
            # return [cv2.arcLength(samples[:i+1],False)  for i in range(len(samples))]/axlength
            if scatter:
                axlength = np.array(cv2.arcLength(samples, False))
                return [cv2.arcLength(samples[:i+1],False)  for i in range(len(samples))]/axlength
            else:
                return [i/(len(samples)-1) for i in range(len(samples))]

    # 制御点のi番目を代入
    def setACP(self, f, i, cp):
        [x, y] = cp
        sx = var('px'+str(i))
        sy = var('py'+str(i))
        f[0] = f[0].subs(sx, x)
        f[1] = f[1].subs(sy, y)
        return f

    # 制御点座標をセットして関数式を完成
    def setCPs(self, cps):
        f = self.f.copy()
        for i in range(self.N+1):
            self.setACP(f, i, cps[i])
        return f

    # ベジエ近似レベル０（標本点のパラメータを等間隔と仮定してあてはめ）
    def fit0(self, prefunc=None, tpara=[], scatter=False, moption=[], withErr=False):
        # ts 標本点に対するパラメータ割り当て  
        samples = self.samples  # 標本点  
        # t 標本点に結びつけるパラメータは引数として与えられているならそれを、さもなくばリニアに設定
        if len(tpara) > 0: #パラメータが与えられているならそれを使う
            ts = self.ts = tpara.copy()
        elif prefunc != None: # 関数が与えられているなら、それをもとに等間隔になるよう初期パラメータ設定
            ts = self.ts = self.assignPara2Samples(prefunc=prefunc) 
        else: # さもなくば、self.prefunc で等間隔か [0,1] を等間隔 ただしscatter=Trueのときは不等間隔
            ts = self.ts = self.assignPara2Samples(prefunc=None,scatter=scatter) 

        N = self.N  # ベジエの次数
        M = len(samples)  # サンプル数
        # バーンスタイン関数の定義
        if len(moption)>1: # 端点部のみ中間点を追加する
            samples = np.concatenate([[samples[0],moption[0]],samples[1:-1],[moption[-1],samples[-1]]])
            ts = np.concatenate([[ts[0],(ts[0]+ts[1])/2],ts[1:-1],[(ts[-2]+ts[-1])/2,ts[-1]]])
            M = M + 2
        x, y = samples[:, 0], samples[:, 1]  # 標本点のｘ座標列とｙ座標列
            
        def bs(n, t):
            return binomial(N, n)*(1-t)**(N-n)*t**n

        # 標本点と、それに対応する曲線上の点の距離の総和を最小化するような制御点を求める
        if BezierCurve.openmode:
            exA = np.array([[sum([bs(i, ts[k])*bs(n, ts[k]) for k in range(M)])
                             for i in range(N+1)] for n in range(N+1)], 'float64')
            if np.linalg.matrix_rank(exA,tol=1e-20) != exA.shape[0]:
                print("Rank Warning(tol:1e-20)")
            exBX = np.array([[sum([x[k]*bs(n, ts[k]) for k in range(M)])]
                             for n in range(N+1)], 'float64')
            exBY = np.array([[sum([y[k]*bs(n, ts[k]) for k in range(M)])]
                             for n in range(N+1)], 'float64')
            cpsx = np.linalg.solve(exA, exBX)
            cpsy = np.linalg.solve(exA, exBY)

        else:  # 両端点をサンプルの両端に固定する場合
            exA = np.array([[sum([bs(i, ts[k])*bs(n, ts[k]) for k in range(M)])
                             for i in range(1, N)] for n in range(1, N)], 'float64')
            if np.linalg.matrix_rank(exA,tol=1e-20) != exA.shape[0]:
                print("Rank Warning(tol:1e-20)")
            exBX = np.array([[sum([bs(n, ts[k])*(x[k]-x[0]*(1-ts[k])**N - x[-1]*ts[k]**N)
                                   for k in range(M)])] for n in range(1, N)], 'float64')
            exBY = np.array([[sum([bs(n, ts[k])*(y[k]-y[0]*(1-ts[k])**N - y[-1]*ts[k]**N)
                                   for k in range(M)])] for n in range(1, N)], 'float64')
            cpsx = np.r_[[[x[0]]], np.linalg.solve(exA, exBX), [[x[-1]]]]
            cpsy = np.r_[[[y[0]]], np.linalg.solve(exA, exBY), [[y[-1]]]]

        cps = [[i[0][0], i[1][0]] for i in zip(cpsx, cpsy)]
        func = self.setCPs(cps)

        if withErr:
            fx,fy = func
            return cps, func, self.f_meanerr(fx, fy, self.ts)
        else:
            return cps, func

    #  パラメトリック曲線　curvefunc 上で各サンプル点に最寄りの点のパラメータを対応づける
    def refineTparaN(self, bezierparameters, curvefunc, stt, end):
        # bezierparameters 標本点と結びつけられたベジエパラメータ
        # curvefunc 曲線の式
        # stt,end パラメータを割り当てる標本番号の最初と最後（最後は含まない）

        sps = self.samples
        ts = bezierparameters
        f = curvefunc

        def searchband(n):
            if n == 0:
                return 0, ts[1]/2.0
            elif n == len(ts)-1:
                return (ts[-2]+1.0)/2.0, 1
            else:
                return (ts[n-1]+ts[n])/2.0, (ts[n]+ts[n+1])/2.0

        # 曲線 linefunc(t) 上で座標(x,y) に最も近い点のパラメータを2分サーチして探す関数 Numpy化で高速化 2021.04.04
        def nearest(x, y, oldt, curvefunc, pmin, pmax, err_th=0.75, dcount=5):
            # x,y 座標、oldt 暫定割り当てのパラメータ、pmin,pmax 探索範囲、dcount 再起呼び出しの残り回数
            # ベジエ曲線の関数を記号式から numpy 関数式に変換

            def us(p):
                x, y = p
                return np.array([float(x), float(y)])

            def nearestNp(p, oldt, funcX, funcY, pmin, pmax, dcount=5):
                # def nearestNp(p, oldt, funcX, funcY, diffX, diffY, pmin, pmax, dcount=5):
                # sympy 表現の座標を数値化
                epsilon = 1e-07
                ps = funcX(pmin), funcY(pmin)  # パラメータ最小点
                pe = funcX(pmax), funcY(pmax)  # パラメータ最大点
                ls = np.linalg.norm(us(ps) - p)  # pmin と p の距離
                le = np.linalg.norm(us(pe) - p)  # pmax と p の距離
                mid = mid = (le*pmin+ls*pmax)/(ls+le)  # pmin と pmax のパラメータの平均
                pm = funcX(mid), funcY(mid)  # 中間パラメータ点
                lold = funcX(oldt)
                lm = np.linalg.norm(us(pm) - p)  # pmid と p の距離
                m = min([ls, lm, le, lold])
                if m == lold:  # 改善されない
                    return oldt
                elif m == lm:
                    newt = mid
                    dd = min(mid - pmin, pmax - mid)/2.0
                    newpmin = mid - dd
                    newpmax = mid + dd
                elif m == le:
                    newt = pmax
                    newpmin = mid  # (mid + pmax)/2.0
                    newpmax = pmax - epsilon
                else:
                    newt = pmin
                    newpmin = pmin + epsilon
                    newpmax = mid  # (pmin + mid)/2.0
                ddx = funcX(newpmax)-funcX(newpmin)  # x の範囲の見積もり
                ddy = funcY(newpmax)-funcY(newpmin)  # y の範囲の見積もり
                ddv = ddx*ddx + ddy*ddy
                if m < err_th or ddv < err_th*err_th or (dcount <= 0 and m > err_th*2):
                    return newt
                else:
                    return nearestNp(p, newt, funcX, funcY, newpmin, newpmax, dcount-1)

            p = np.array([x, y])
            t = symbols('t')
            (funcX, funcY) = curvefunc  # funcX,funcY は 't' の関数
            # (diffX, diffY) = (diff(funcX, 't'), diff(funcY, 't'))  # 導関数
            # 関数と導関数を numpy 関数化　（高速化目的）
            (funcX, funcY) = (lambdify(t, funcX, "numpy"), lambdify(t, funcY, "numpy"))
            # (diffX, diffY) = (lambdify(t, diffX, "numpy"), lambdify(t, diffY, "numpy"))
            # return nearestNp(p, oldt, funcX, funcY, diffX, diffY, pmin, pmax, dcount=dcount)
            return nearestNp(p, oldt, funcX, funcY, pmin, pmax, dcount=dcount)

        if stt == end:
            return ts

        nmid = (stt+end)//2  # 探索対象の中央のデータを抜き出す
        px, py = sps[nmid]  # 中央のデータの座標
        band = searchband(nmid)
        tmid = ts[nmid]

        midpara = nearest(
            px, py, tmid, f, band[0], band[1], dcount=BezierCurve.dCount)  # 最も近い点を探す

        ts[nmid] = midpara
        ts = self.refineTparaN(ts, f, stt, nmid)
        ts = self.refineTparaN(ts, f, nmid+1, end)

        return ts

    # ベジエ近似　パラメータの繰り返し再調整あり
    def fit1(self, maxTry=0, withErr=False, withEC=False, tpara=[], prefunc=None,pat=300, err_th=0.75, threstune=1.00,scatter=False, moption=[]):
        # maxTry 繰り返し回数指定　0 なら誤差条件による繰り返し停止
        # withErr 誤差情報を返すかどうか  withECも真ならカウントも返す
        # tpara  fit0() にわたす初期パラメータ値
        # prefunc tpara を求める基準となる関数式がある場合は指定
        # pat 300 これで指定する回数最小エラーが更新されなかったら繰り返しを打ち切る
        # threstune 1.0  100回以上繰り返しても収束しないとき、この割合で収束条件を緩める
        # scatter 不等間隔標本の場合に真
        start = time.process_time()
        flag5,flag2,flag1,flag65,flag05=True,True,True,True,True
        
        sps = self.samples

        # #######################
        # Itterations start here フィッティングのメインプログラム
        # #######################
        trynum = 0  # 繰り返し回数
        lastgood = -1
        rmcounter = 0  # エラー増加回数のカウンター
        priority = BezierCurve.AsymptoticPriority

        N = self.N
        # 初期の仮パラメータを決めるため、fit0(2N)で近似してみる 
        doubleN = 2*N if N < 9 else 18
        prebez = BezierCurve(N=doubleN,samples=self.samples)
        precps, prefunc = prebez.fit0(prefunc=prefunc,tpara=tpara,scatter=scatter, moption=moption)
        # 仮近似曲線をほぼ等距離に区切るようなパラメータを求める
        # 改めて fit0 でN次近似した関数を初期近似とする
        # cps, func = self.fit0(prefunc = prefunc, tpara=tpara)  # レベル０フィッティングを実行
        cps, func = self.fit0(prefunc = prefunc, scatter=scatter, moption=moption)  # レベル０フィッティングを実行

        #cps, func = self.fit0(tpara=tpara,scatter=scatter, moption=moption)  # レベル０フィッティングを実行
        [fx, fy] = bestfunc = func
        bestcps = cps
        ts = bestts = self.ts.copy()

        minerror = self.f_meanerr(fx, fy, ts=ts)  # 当てはめ誤差
        errq = deque(maxlen=3) # エラーを３回分記録するためのバッファ
        for i in range(2):
          errq.append(np.inf)
        errq.append(minerror)
          
        if BezierCurve.debugmode:
            print("initial error:{:.5f}".format(minerror))

        while True:
            # パラメータの再構成（各標本点に関連付けられたパラメータをその時点の近似曲線について最適化する）
            if priority == 'distance' or priority == 'hyblid':
                ts = self.refineTparaN(ts, [fx, fy], 0, len(sps))
            # 標本点が等間隔であることを重視し、曲線上の対応点も等間隔であるということを評価尺度とする方法
            elif priority == 'span':
                ts = self.assignPara2Samples(prefunc=[fx, fy])

            # レベル０フィッティングを再実行
            cps, func = self.fit0(tpara=ts,scatter=scatter,moption=moption)
            [fx, fy] = func

            # あてはめ誤差を求める
            error = self.f_meanerr(fx, fy, ts=ts)
            old3err = errq.popleft() # ３回前のエラーを取り出し
            errq.append(error) # 最新エラーをバッファに挿入

            if BezierCurve.wandb:
                BezierCurve.wandb.log({"loss": error})
            if error < minerror:
                convg_coe = BezierCurve.convg_coe
                convergenceflag = (trynum - lastgood > 3 or trynum - lastgood == 1) and ((old3err - error)/3.0 < convg_coe*(error-err_th))
                # convergenceflag = True if (minerror - error)/(trynum - lastgood) < convg_coe*(error-err_th) else False
                lastgood = trynum
                bestts = ts.copy()  # 今までで一番よかったパラメータセットを更新
                bestfunc = func  # 今までで一番よかった関数式を更新
                minerror = error  # 最小誤差を更新
                bestcps = cps  # 最適制御点リストを更新
                print(".", end='')
            else:
                print("^", end='')

            # 繰り返し判定調整量
            # 繰り返しが100回を超えたら条件を緩めていく
            thresrate = 1.0 if trynum <= 100 else threstune**(trynum-100)
            if BezierCurve.debugmode:
                print("{} err:{:.5f}({:.5f}) rmcounter {})".format(
                    trynum, error, minerror, rmcounter))

            rmcounter = 0 if error <= minerror else rmcounter + 1  # エラー増加回数のカウントアップ　減り続けているなら０
            # if convergenceflag or error < err_th*thresrate or rmcounter > pat:
            if (convergenceflag and (trynum > 50)) or error < err_th*thresrate or ((trynum > 100) and rmcounter > pat):
                # pat回続けてエラーが増加したらあきらめる デフォルトは10
                #if (convergenceflag and (trynum > 50)):
                if convergenceflag:
                  print('C')
                elif error < err_th*thresrate :
                  print('E')
                else:
                  print('P')
                if BezierCurve.debugmode:
                    if rmcounter > pat:
                        print("W")
                    else:
                        print("M")
                if priority == 'hyblid':
                    rmcounter = 0
                    priority = 'span'
                else:
                    break
                
            if error < 5.0 and flag5:
                etime = time.process_time() - start
                print("\nCP 5.0, steps:%d, etime: %3.5f err: %1.10f" % (trynum,etime, error))
                flag5 = False
            elif  error < 2.0 and flag2:
                etime = time.process_time() - start
                print("\nCP 2.0, steps:%d, etime: %3.5f err: %1.10f" % (trynum,etime, error))
                flag2 = False
            elif  error < 1.0 and flag1:
                etime = time.process_time() - start
                print("\nCP 1.0, steps:%d, etime: %3.5f err: %1.10f" % (trynum,etime, error))
                flag1 = False
            elif  error < 0.65 and flag65:
                etime = time.process_time() - start
                print("\nCP 0.65, steps:%d, etime: %3.5f err: %1.10f" % (trynum,etime, error))
                flag65 = False
            elif error < 0.5 and flag05:
                etime = time.process_time() - start
                print("\nCP 0.5, steps:%d, etime: %3.5f err: %1.10f" % (trynum,etime, error))
                flag05 = False
                
            trynum += 1
            if trynum % 100 == 0:
                print("")
            if maxTry > 0 and trynum >= maxTry:
                break

        self.ts = bestts.copy()
        
        print("")
        if withErr:
            if withEC:
                return bestcps, bestfunc, (minerror,trynum)
            else:
                return bestcps, bestfunc, minerror
        else:
            return bestcps, bestfunc

    # fit1 の tensorflowによる実装
    def fit1T(self, mode=1, maxTry=0, withErr=False, withEC=False,prefunc=None,tpara=[], lr=0,  lrP=0, pat=300, err_th=0.75, threstune=1.00, trial=None, scatter=False, moption=[]):
        # maxTry 繰り返し回数指定　0 なら誤差条件による繰り返し停止
        # withErr 誤差情報を返すかどうか withEC が真ならカウントも返す
        # tpara  fit0() にわたす初期パラメータ値
        # mode 0: 制御点とパラメータを両方同時に tensorflow で最適化する
        # mode 1: パラメータの最適化は tensorflow で、制御点はパラメータを固定して未定係数法で解く
        # fit1T では priority=distance のみ考え、span は考慮しない
        # lr オプティマイザーの学習係数（媒介変数 ts 用）
        # lrP 制御点用オプティマイザーの学習係数の倍率 lr*lrP を制御点の学習係数とする。
        # prefunc tpara を求める基準となる関数式がある場合は指定
        # pat 10 これで指定する回数最小エラーが更新されなかったら繰り返しを打ち切る
        # err_th 0.75  エラーの収束条件
        # threstune 1.0  100回以上繰り返しても収束しないとき、この割合で収束条件を緩める
        # trial Optuna のインスタンス
        # サンプル間隔が等間隔でない場合、scatter=True
        start = time.process_time()
        flag5,flag2,flag1,flag65,flag05=True,True,True,True,True
        
        sps = self.samples.copy()
        
        # #######################
        # Itterations start here フィッティングのメインプログラム
        # #######################
        trynum = 0  # 繰り返し回数
        lastgood = -1
        rmcounter = 0  # エラー増加回数のカウンター
        priority = BezierCurve.AsymptoticPriority

        N = self.N
        # 初期の仮パラメータを決めるため、fit0(2N)で近似してみる 
        doubleN = 2*N  if N < 9 else 18
        prebez = BezierCurve(N=doubleN,samples=self.samples)
        precps, prefunc = prebez.fit0(prefunc=prefunc, tpara=tpara,scatter=scatter, moption=moption)
        # 仮近似曲線をほぼ等距離に区切るようなパラメータを求める
        # 改めて fit0 でN次近似した関数を初期近似とする
        # cps, func = self.fit0(prefunc = prefunc, tpara=tpara)  # レベル０フィッティングを実行
        cps, func = self.fit0(prefunc = prefunc, scatter=scatter, moption=moption)  # レベル０フィッティングを実行
        (fx,fy) = bestfunc = func
        bestcps = cps
        ts = bestts = self.ts.copy()
        m_ts = self.ts.copy()
        if len(moption)>1: # 端点部のみ中間点を追加する
            sps = np.concatenate([[sps[0],moption[0]],sps[1:-1],[moption[-1],sps[-1]]])
            m_ts = np.concatenate([[m_ts[0],(m_ts[0]+m_ts[1])/2],m_ts[1:-1],[(m_ts[-2]+m_ts[-1])/2,m_ts[-1]]])
        x_data = sps[:,0]
        y_data = sps[:,1]

        minerror = self.f_meanerr(fx, fy, ts=ts)  # 初期当てはめ誤差の算出
        errq = deque(maxlen=3) # エラーを３回分記録するためのバッファ
        for i in range(2):
          errq.append(np.inf)
        errq.append(minerror)

        if BezierCurve.debugmode:
            print("initial error:{:.5f}".format(minerror))

        # tensorflow の変数
        if mode == 0:
            Px = tf.Variable([cps[i+1][0]
                              for i in range(N-1)], dtype='float32')
            Py = tf.Variable([cps[i+1][1]
                              for i in range(N-1)], dtype='float32')

        tts = tf.Variable(m_ts, dtype='float32') # 端点以外のベジエパラメータをtensorflow変数化
        
        # Optimizer 
        # 実験結果としてAdamが優れていたのでAdamを採用。Adam 以外の、経験的な適正パラメータは以下の通り
        # 'AMSgrad':[0.005,650],'Adagrad':[0.02,250],
        # 'Adadelta':[0.013,3500],'Nadam':[0.001,500], 'Adamax':[0.0075,1000],
        # 'RMSprop':[0.0003,1300],'SGD':[5e-6,2e6],'Ftrl':[0.12,1000]}
        default_lrs={'Adam':[0.0015,1140]}        
        if lr == 0: lr = default_lrs['Adam'][0]
        if lrP == 0: lrP = default_lrs['Adam'][1]
        opt = tf.optimizers.Adam(learning_rate=lr) 
        optP = tf.optimizers.Adam(learning_rate=lr*lrP) 

        tfZERO = tf.constant(0.0,tf.float32)
        tfONE = tf.constant(1.0,dtype=tf.float32)
        
        while True:
            olderror = self.f_meanerr(fx, fy, ts=ts) 
            for loopc in range(BezierCurve.mloop_itt):
                # パラメータの再構成（各標本点に関連付けられたパラメータをその時点の近似曲線について最適化する）
                # 関数化したかったが、tape をつかっているせいなのか、エラーがでてできなかった
                with tf.GradientTape(persistent=True) as metatape:
                    vs = tfONE-tts
                    # tts**0 と (1-tts)**0 を含めると微係数が nan となるのでループから外している
                    bezfx = vs**N*cps[0][0]
                    bezfy = vs**N*cps[0][1] 
                    for i in range(1, N):
                        if mode == 0:
                            bezfx = bezfx + comb(N, i)*vs**(N-i)*tts**i*Px[i-1]
                            bezfy = bezfy + comb(N, i)*vs**(N-i)*tts**i*Py[i-1]                                
                        elif mode == 1:
                            bezfx = bezfx + comb(N, i)*vs**(N-i)*tts**i*cps[i][0]
                            bezfy = bezfy + comb(N, i)*vs**(N-i)*tts**i*cps[i][1]
                    bezfx = bezfx + tts**N*cps[N][0]
                    bezfy = bezfy + tts**N*cps[N][1]
                    
                    sqx,sqy = tf.square(bezfx - x_data),tf.square(bezfy - y_data)

                    meanerrx = tf.reduce_mean(tf.square(bezfx - x_data))
                    meanerry = tf.reduce_mean(tf.square(bezfy - y_data))
                    gloss = tf.add(meanerrx, meanerry)                                      

                # ts を誤差逆伝搬で更新
                opt.minimize(gloss, tape=metatape, var_list=tts)
                # ts を更新
                m_ts = tts.numpy() 
                m_ts[0] = 0.0 # 両端は0,1にリセット
                m_ts[-1] = 1.0 
                # check order and reorder 順序関係がおかしい場合強制的に変更       
                ec = 0
                for i in range(1,len(m_ts)-1):
                    if m_ts[i] <= m_ts[i-1]:
                        ec += 1
                        m_ts[i] = m_ts[i-1] + 1e-6
                    if m_ts[i] >= 1.0-(len(m_ts)-i-2)*(1e-6):
                        ec += 1
                        m_ts[i] = min(1.0-(len(m_ts)-i-2)*(1e-6), m_ts[i+1]) - 1e-6
                if ec > 0:
                    print("e%d" % (ec),end="")
            # tts を更新
            tts.assign(m_ts)
            if len(moption)>0:
                ts[1:-1] = m_ts[2:-2].copy()
            else:
                ts = m_ts.copy()
            self.ts = ts.copy()

            if mode == 0:
                # 上で求めたベジエパラメータに対し制御点を最適化
                for loopc in range(BezierCurve.mloop_itt):
                    with tf.GradientTape(persistent=True) as metatape:
                        vs = tfONE-tts
                        # tts**0 と (1-tts)**0 を含めると微係数が nan となるのでループから外している
                        bezfx = vs**N*cps[0][0]
                        bezfy = vs**N*cps[0][1]
                        for i in range(1, N):
                            bezfx = bezfx + comb(N, i)*vs**(N-i)*tts**i*Px[i-1]
                            bezfy = bezfy + comb(N, i)*vs**(N-i)*tts**i*Py[i-1]
                        bezfx = bezfx + tts**N*cps[N][0]
                        bezfy = bezfy + tts**N*cps[N][1]

                        meanerrx = tf.reduce_mean(tf.square(bezfx - x_data))
                        meanerry = tf.reduce_mean(tf.square(bezfy - y_data))
                        gloss = tf.add(meanerrx, meanerry)                

                    optP.minimize(gloss, tape=metatape, var_list=[Px, Py])

                for i in range(1, N):
                    cps[i][0] = Px[i-1].numpy()
                    cps[i][1] = Py[i-1].numpy() 
                func = self.setCPs(cps)
            elif mode == 1:
                cps, func = self.fit0(tpara=ts,scatter=scatter,moption=moption)

            # 誤差評価
            fx,fy = func
            error = self.f_meanerr(fx, fy, ts=ts) 
            old3err = errq.popleft() # ３回前のエラーを取り出し
            errq.append(error) # 最新エラーをバッファに挿入
            convg_coe = BezierCurve.convg_coe # if mode == 1 else BezierCurve.convg_coe/10.0 # 収束判定基準
            convergenceflag = (trynum - lastgood > 3 or trynum - lastgood == 1) and ((old3err - error)/3.0 < convg_coe*(error-err_th))
            
            if BezierCurve.wandb:
                BezierCurve.wandb.log({"loss":error})
            if error < minerror: 
                if trynum - lastgood > 3:
                  convergenceflag = False
                bestts = ts.copy()  # 今までで一番よかったパラメータセットを更新
                bestfunc = func  # 今までで一番よかった関数式を更新
                minerror = error  # 最小誤差を更新
                bestcps = cps  # 最適制御点リストを更新
                print(".", end='')                  
                lastgood = trynum
                rmcounter = 0
            else:
                rmcounter = rmcounter + 1 # エラー増加回数のカウントアップ　減り続けているなら０
                convergenceflag = False
                print("^", end='') 

            # 繰り返しが100回を超えたら条件を緩めていく
            thresrate = 1.0 if trynum <= 100 else threstune**(trynum-100)
            if BezierCurve.debugmode:
                print("{} err:{:.5f}({:.5f}) rmcounter {})".format(
                    trynum, error, minerror, rmcounter))
            
            # エラーが増加したときだけ fit0 を実行
            # if convergenceflag or errq[-2] < error:
                # fit0 で近似し、間隔均等になるように初期パラメータを決定
                # cps, func = self.fit0(tpara=tts.numpy())
                # ts = self.assignPara2Samples(prefunc=func)

            if (convergenceflag and (trynum > 50)) or error < err_th*thresrate or ((trynum > 100) and rmcounter > pat):
                # pat回続けてエラーが増加したらあきらめる デフォルトは１00 （fit1T は繰り返し1回あたりの変動が小さい）
                if (convergenceflag and (trynum > 50)):
                  print('C')
                elif error < err_th*thresrate :
                  print('E')
                else:
                  print('P')

                if BezierCurve.debugmode:
                    if rmcounter > pat:
                        print("W")
                    else:
                        print("M")
                break

            if error < 5.0 and flag5:
                etime = time.process_time() - start
                print("\nCP 5.0, steps:%d, etime: %3.5f err: %1.10f" % (trynum,etime, error))
                flag5 = False
            elif  error < 2.0 and flag2:
                etime = time.process_time() - start
                print("\nCP 2.0, steps:%d, etime: %3.5f err: %1.10f" % (trynum,etime, error))
                flag2 = False
            elif  error < 1.0 and flag1:
                etime = time.process_time() - start
                print("\nCP 1.0, steps:%d, etime: %3.5f err: %1.10f" % (trynum,etime, error))
                flag1 = False
            elif  error < 0.65 and flag65:
                etime = time.process_time() - start
                print("\nCP 0.65, steps:%d, etime: %3.5f err: %1.10f" % (trynum,etime, error))
                flag65 = False
            elif error < 0.5 and flag05:
                etime = time.process_time() - start
                print("\nCP 0.5, steps:%d, etime: %3.5f err: %1.10f" % (trynum,etime, error))
                flag05 = False
                      
            trynum += 1
            if trynum % 100 == 0:
                print("")
            if maxTry > 0 and trynum >= maxTry:
                break
            # Optuna を使っている場合の打ち切り
            if trial:
                trial.report(error,trynum)
                if trial.should_prune() or error > 1e3:
                    print("Optuna による打ち切り")
                    raise optuna.TrialPruned()
            
        self.ts = bestts.copy()
        fx,fy=bestfunc

        if withErr:
            if withEC:
                return bestcps, bestfunc, (minerror,trynum) 
            else:
                return bestcps, bestfunc, minerror
        else:
            return bestcps, bestfunc

    # 段階的ベジエ近似
    def fit2(self, mode=0, cont = [], Nprolog=0, Nfrom=5, Nto=12, preTry=200, maxTry=0, lr=0.001, lrP=30000, pat=300, err_th=0.75, threstune=1.0, withErr=False, withEC=False, tpara=[], withFig=False, moption=[]):
        # mode 0 -> fit1() を使う, mode 1 -> fit1T(mode=1)を使う, mode 2 -> fit1T(mode=0) を使う
        # contours 与えられている場合オーバフィッティング判定を行う
        # Nplolog 近似準備開始次数　この次数からNfrom-1までは maxTry 回数で打ち切る
        # Nfrom 近似開始次数　この次数以降は収束したら終了
        # Nto 最大近似次数 Nto < Nfrom  の場合は誤差しきい値による打ち切り
        # maxTry 各次数での繰り返し回数
        # prefunc 初期近似関数
        # err_th 打ち切り誤差
        # pat この回数エラーが減らない場合はあきらめる
        # withErr 誤差と次数を返すかどうか

        Nprolog = Nfrom if Nprolog == 0 else Nfrom
        Ncurrent = Nprolog - 1
        func = self.prefunc
        ts = tpara
        err = err_th + 1
        results = {}
        odds = []
        while Ncurrent < Nto and (err_th < err or len(odds) > 0):
            Ncurrent = Ncurrent + 1
            # abez = BezierCurve(N=Ncurrent, samples=self.samples, tpara=ts, prefunc=func)
            abez = BezierCurve(N=Ncurrent, samples=self.samples, tpara=[], prefunc=None)
            print(Ncurrent, end="")
            # 最大 maxTry 回あてはめを繰り返す
            if mode == 0:
                cps, func, err = abez.fit1(
                    maxTry=preTry if Ncurrent < Nfrom else maxTry, withErr=True, tpara=[], pat=pat, err_th=err_th, threstune=threstune, moption=moption)
            elif mode == 1: # XXXX
                cps, func, err = abez.fit1T(
                    mode=1, maxTry=preTry if Ncurrent < Nfrom else maxTry, lr=lr, lrP=lrP,withErr=True, tpara=[], pat=pat, err_th=err_th, threstune=threstune, moption=moption)
            elif mode == 2:
                cps, func, err = abez.fit1T(
                    mode=0, maxTry=preTry if Ncurrent < Nfrom else maxTry, lr=lr, lrP=lrP,withErr=True, tpara=[], pat=pat, err_th=err_th, threstune=threstune, moption=moption)
            ts = abez.ts
            if len(cont)>0:
                odds = isOverFitting(func=func,ts=ts,cont=cont,err_th=err_th,of_th=1.0) 
            if err_th >= err and len(odds) > 0:
                print("Order ",Ncurrent," is Overfitting",odds)
            results[str(Ncurrent)] = (cps, func, err)
            # 次数を上げてインスタンス生成
        self.ts = ts.copy()
        print(err,end="")
        if withErr:
            return Ncurrent, results
        else:
            return cps, func

    # デバッグモードのオンオフ
    def toggledebugmode(set=True, debug=False):
        if set:
            BezierCurve.debugmode = debug
        else:  # set が False のときはトグル反応
            BezierCurve.debugmode = not BezierCurve.debugmode
        print("debugmode:", BezierCurve.debugmode)

    # パラメータのセットと表示　引数なしで呼ぶ出せば初期化
    def setParameters(priority='distance', dCount=3, convg_coe=1e-5,swing_penalty=0.0,smoothness_coe=0.0, debugmode=False, openmode=False,wandb=None):

        BezierCurve.AsymptoticPriority = priority  # パラメータ割り当てフェーズにおける評価尺度

        # xx BezierCurve.driftThres = driftThres # 繰り返しにおけるパラメータ変動幅の平均値に対するしきい値
        # xx BezierCurve.errorThres = errorThres # 繰り返しにおける誤差変動幅に対するしきい値
        BezierCurve.dCount = dCount  # サンプル点の最寄り点の2分探索の回数
        BezierCurve.debugmode = debugmode
        BezierCurve.openmode = openmode
        BezierCurve.wandb = wandb
        BezierCurve.convg_coe = convg_coe
        BezierCurve.swing_penalty = swing_penalty
        BezierCurve.smoothness_coe = smoothness_coe
        print("AsymptoticPriority : ", priority)
        print("dCount    : ", dCount)
        print("debugmode : ", debugmode)
        print("openmode  : ", openmode)
        print("wandb  : ", wandb)
        print("convg_coe :", convg_coe)
        print("swing_penalty :", swing_penalty)
        print("smoothness_coe :", smoothness_coe)
        print("")

# (28) カラーの名前をからコードに


def n2c(name):
    cmap = plt.get_cmap("tab10")  # カラーマップ
    # 0:darkblue,1:orange,2:green,3:red,4:purple,
    # 5:brown,6:lpurple,7:gray,8:leaf,9:rikyu
    # 0#1f77b4:1#ff7f0e:2#2ca02c:3#d62728:4#9467bd
    # 5#8c564b:6#e377c2:7#7f7f7f:8#bcbd22:9#17becf
    # https://www.color-sample.com/
    c = {'blue': 0, 'orange': 1, 'green': 2, 'red': 3, 'purple': 4,
         'brown': 5, 'lpurple': 6, 'gray': 7, 'leaf': 8, 'rikyugreen': 9}
    if type(name) == str:
        if len(name) > 1:
            return cmap(c[name])
        else:
            return cmap(int(name))
    elif type(name) == int:
        return cmap(name)
    else:
        return cmap(0)

# (29) ベジエフィッティングの結果の描画


def drawBez(rdimg, stt=0.02, end=0.98, bezL=None, bezR=None, bezC=None, cpl=[], cpr=[], cpc=[],
            cntL=[], cntR=[], cntC=[], ladder=None, PosL=[], PosR=[], PosC=[], saveImage=False, savepath="",
            resolution=128, n_ladder=20, ct=['red', 'red', 'red', 'blue', 'blue', 'blue', 'purple', 'red', 'rikyugreen', 'orange'],
            figsize=(6, 6), dpi=100, layout=111, bzlabel="", linestyle='solid'):

    # rdimg 入力画像、stt,end 曲線の描画範囲、
    # bezL,bezR,bezC ベジエ曲線、cpl,cpr,cpc 制御点
    # cntL,cntR,cntC 標本点のリスト,
    # ladder 梯子描画モード、
    # PosL,PosR,PosC ラダー用座標
    # saveImage 結果の保存の有無
    # resolution 曲線を描画する際に生成する描画点の数

    if figsize != None:
        plt.figure(figsize=figsize, dpi=dpi)

    plt.subplot(layout)

    drawBez0(rdimg, stt=stt, end=end, bezL=bezL, bezR=bezR, bezC=bezC, cpl=cpl, cpr=cpr, cpc=cpc,
             cntL=cntL, cntR=cntR, cntC=cntC, ladder=ladder, PosL=PosL, PosR=PosR, PosC=PosC, saveImage=saveImage, savepath=savepath,
             resolution=resolution, n_ladder=n_ladder, ct=ct, bzlabel=bzlabel, linestyle=linestyle)

# (29)-2 # 重ね書き用

def drawBez0(rdimg, stt=0.02, end=0.98, bezL=None, bezR=None, bezC=None, cpl=[], cpr=[], cpc=[],
             cntL=[], cntR=[], cntC=[], ladder=None, PosL=[], PosR=[], PosC=[], saveImage=False, savepath="",
             resolution=128, n_ladder=20, ct=['red', 'red', 'red', 'blue', 'blue', 'blue', 'purple', 'red', 'rikyugreen', 'orange'],
             bzlabel="", linestyle='solid'):
    if type(linestyle) == int:
        if 0 < linestyle and linestyle < 4:
            linestyle = ["solid", "dashed", "dashdot", "dotted"][linestyle]
        else:
            linestyle = 'solid'
    # matplotlib の仕様変更への対応 数字列指定した場合
    if ct[0].isdecimal() == True:
        ct = [n2c(int(i)) for i in ct]
    # いわゆる自乗誤差の一般式
    s, t = symbols('s,t')

    bezXl, bezYl = bezL if bezL != None else ([], [])
    bezXr, bezYr = bezR if bezR != None else ([], [])
    bezXc, bezYc = bezC if bezC != None else ([], [])
    cpxl, cpyl = [x for [x, y] in cpl], [
        y for [x, y] in cpl] if len(cpl) > 0 else ([], [])
    cpxr, cpyr = [x for [x, y] in cpr], [
        y for [x, y] in cpr] if len(cpr) > 0 else ([], [])
    cpxc, cpyc = [x for [x, y] in cpc], [
        y for [x, y] in cpc] if len(cpc) > 0 else ([], [])
    tplins50 = np.linspace(stt, end, resolution)
    tplinsSP = np.linspace(stt, end, n_ladder)

    # plt.figure(figsize=(6,6),dpi=100)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')  # アスペクト比を１：１に
    plt.imshow(192+(cv2.cvtColor(rdimg, cv2.COLOR_GRAY2RGB)/4).astype(np.uint8))

    # 左輪郭の描画
    if bezL != None:
        if len(cntL) > 0:
            tplins50 = np.linspace(stt, end, 5*len(cntL))
        nbezXl, nbezYl = lambdify(
            t, bezXl, "numpy"), lambdify(t, bezYl, "numpy")
        plotx = [nbezXl(tp) for tp in tplins50]
        ploty = [nbezYl(tp) for tp in tplins50]
        plt.plot(plotx, ploty, color=ct[0], label=bzlabel, linestyle=linestyle)  # red
    if len(cntL) > 0:
        plt.scatter(cntL[:, 0], cntL[:, 1], color=ct[3], marker='.')  # サンプル点 blue
    if len(cpl) > 0:  # 制御点
        plt.scatter(cpxl, cpyl, color=ct[6], marker='*')  # 制御点の描画 purple
        for i in range(len(cpxl)):
            plt.annotate(str(i), (cpxl[i], cpyl[i]))
    # 右輪郭の描画
    if bezR != None:
        if len(cntR) > 0:
            tplins50 = np.linspace(stt, end, 5*len(cntR))
        nbezXr, nbezYr = lambdify(
            t, bezXr, "numpy"), lambdify(t, bezYr, "numpy")
        plotx = [nbezXr(tp) for tp in tplins50]
        ploty = [nbezYr(tp) for tp in tplins50]
        plt.plot(plotx, ploty, color=ct[1], linestyle=linestyle)  # red
    if len(cntR) > 0:
        plt.scatter(cntR[:, 0], cntR[:, 1], color=ct[4], marker='.')  # サンプル点 blue
    if len(cpr) > 0:
        plt.scatter(cpxr, cpyr, color=ct[7], marker='*')  # 制御点の描画 red
        for i in range(len(cpxr)):
            plt.annotate(str(i), (cpxr[i], cpyr[i]))
    # 中心軸の描画
    if bezC != None:
        if len(cntC) > 0:
            tplins50 = np.linspace(stt, end, 5*len(cntC))
        nbezXc, nbezYc = lambdify(
            t, bezXc, "numpy"), lambdify(t, bezYc, "numpy")
        plotx = [nbezXc(tp) for tp in tplins50]
        ploty = [nbezYc(tp) for tp in tplins50]
        plt.plot(plotx, ploty, color=ct[2], linestyle=linestyle)  # red
        if len(cntC) > 0:
            plt.scatter(cntC[:, 0], cntC[:, 1], color=ct[5], marker='.')  # サンプル点 blue
        if len(cpc) > 0:
            plt.scatter(cpxc, cpyc, color=ct[8], marker='*')  # 制御点の描画 rikyugreen
            for i in range(len(cpxc)):
                plt.annotate(str(i), (cpxc[i], cpyc[i]))

        # ラダーの描画
        if ladder == 'lr':  # 左右の同じパラメータ値の点を結ぶだけ
            plotSPlx = [nbezXl(tp) for tp in tplinsSP]
            plotSPly = [nbezYl(tp) for tp in tplinsSP]
            plotSPrx = [nbezXr(tp) for tp in tplinsSP]
            plotSPry = [nbezYr(tp) for tp in tplinsSP]
            for x0, x1, y0, y1 in zip(plotSPlx, plotSPrx, plotSPly, plotSPry):
                plt.plot([x0, x1], [y0, y1], color=ct[9])  # orange

        elif ladder == 'normal':
            # 中心軸上に設定したサンプル点における法線と両輪郭の交点のリストを求める。
            plot20lx = PosL[:, 0]
            plot20ly = PosL[:, 1]
            plot20cx = PosC[:, 0]
            plot20cy = PosC[:, 1]
            plot20rx = PosR[:, 0]
            plot20ry = PosR[:, 1]
            for x0, x1, y0, y1 in zip(plot20lx, plot20cx, plot20ly, plot20cy):
                if x0 != np.inf and y0 != np.inf:
                    plt.plot([x0, x1], [y0, y1], color=ct[9])  # orange
            for x0, x1, y0, y1 in zip(plot20rx, plot20cx, plot20ry, plot20cy):
                if x0 != np.inf and y0 != np.inf:
                    plt.plot([x0, x1], [y0, y1], color=ct[9])  # orange
    if saveImage:
        pltsaveimage(savepath, 'Bez')

# (30) matplotlib で描いた画像の保存


def pltsaveimage(savepath, prefix):
    # 結果を保存する
    savedir, filename = os.path.split(savepath)
    #  _,subdir = os.path.split(savedir)
    os.makedirs(savedir, exist_ok=True)  # 保存先フォルダがなければ作成
    savepath = os.path.join(savedir, prefix+filename)
    if os.path.exists(savepath):
        os.remove(savepath)
    print("TEST", savepath)
    plt.savefig(savepath)

# (31) 画像の両側と仮の中心線のベジエ曲線を返す関数


def getAverageBezline(img, N=6, n_samples=32, Amode=0, maxTry=0, moption=True): # ここはTrueで正しい
    # img 画像
    # N ベジエ近似の次数
    # n_samples 左右輪郭線から取るサンプル点の数
    # Amode 近似方法 N006参照
    # maxTry Amode=1のときの、最大繰り返し回数
    # OverFitting 対策実施　ここは[]でなくTrueなので注意

    # 左右の輪郭を抽出
    conLeft, conRight = getCntPairWithImg(img, dtopdr=1, dbtmdr=1)
    # 輪郭点を（チェインの並び順に）等間隔に n_samples 個サンプリングする。
    # 左右の輪郭点をベジエ近似する
    cntL = getSamples(conLeft, N=n_samples, mode='Equidistant')
    cntR = getSamples(conRight, N=n_samples, mode='Equidistant')

    # ベジエ曲線のインスタンスを生成
    bezL = BezierCurve(N=N, samples=cntL)
    bezR = BezierCurve(N=N, samples=cntR)

    # 左右をそれぞれベジエ 曲線で近似し、その平均として中心軸を仮決定
    if Amode == 0:
        if moption:
            cpl, fL = bezL.fit0(moption=getMutPoints(conLeft,cntL))
            cpr, fR = bezR.fit0(moption=getMutPoints(conRight,cntR))
        else:
            cpl, fL = bezL.fit0(moption=[])
            cpr, fR = bezR.fit0(moption=[])            
    else:
        if moption:
            cpl, fL = bezL.fit1T(moption=getMutPoints(conLeft,cntL))
            cpr, fR = bezR.fit1T(moption=getMutPoints(conRight,cntR))
        else:
            cpl, fL = bezL.fit1T(moption=[])
            cpr, fR = bezR.fit1T(moption=[]) 

    fC = (fL+fR)/2
    cpc = [x for x in (np.array(cpl)+np.array(cpr))/2]
    return cpl, cpr, cpc, fL, fR, fC, cntL, cntR

# (32) 中心線の法線と輪郭の交点を数式処理により求める


def crossPointsLRonEx(fl, fr, fc, t0):
    # fl,fr,fc 左側、右側、中心線のパラメトリック曲線
    # t0 曲線上の位置を特定するパラメータ
    t = symbols('t')
    fcx, fcy = fc
    flx, fly = fl
    frx, fry = fr
    dcx, dcy = diff(fcx, t), diff(fcy, t)
    x0 = float(fcx.subs(t, t0))  # (x0,y0) 中心軸上の点
    y0 = float(fcy.subs(t, t0))
    dx0 = float(dcx.subs(t, t0))  # dx/dt
    dy0 = float(dcy.subs(t, t0))  # dy/dt

    ans = solve(-dx0/dy0*(frx-x0) + y0-fry, t)  # 法線とベジエ輪郭の交点を求める
    ansR = [re(i) for i in ans if float(Abs(im(i))) < 0.00000001]
    # ↑理論的には、im(i) == 0  でいいのだが、数値計算誤差で虚部が０とならず、微小な値となる現象に現実的な対応
    sr = [i for i in ansR if i <= 1.03 and -0.03 <= i]  # ０から１までの範囲の解を抽出
    rdata = [int(float(frx.subs(t, sr[0]))), int(
        float(fry.subs(t, sr[0])))] if sr != [] else [np.inf, np.inf]

    ans = solve(-dx0/dy0*(flx-x0)+y0-fly, t)  # 法線とベジエ輪郭の交点を求める
    ansL = [re(i) for i in ans if float(Abs(im(i))) < 0.00000001]
    sl = [i for i in ansL if i <= 1.03 and -0.03 <= i]
    ldata = [int(float(flx.subs(t, sl[0]))), int(
        float(fly.subs(t, sl[0])))] if sl != [] else [np.inf, np.inf]

    return ldata, rdata

# (33) 中心線の法線と輪郭の交点を図的処理により求める


def crossPointsLRonImg(img, fc, t0, debugmode=False):
    # fc 中心線のパラメトリック曲線
    # t0 曲線上の位置を特定するパラメータ
    t = symbols('t')
    fcx, fcy = fc
    dcx, dcy = diff(fcx, t), diff(fcy, t)
    x0 = int(float(fcx.subs(t, t0)))  # (x0,y0) 中心軸上の点
    y0 = int(float(fcy.subs(t, t0)))
    dx0 = float(dcx.subs(t, t0))  # dx/dt
    dy0 = float(dcy.subs(t, t0))  # dy/dt

    # 法線と輪郭の交点を図的に求める　
    (crpLx, crpLy), (crpRx, crpRy) = crossPointsLRonImg0(img, x0, y0, dx0, dy0)

    # 異常判定
    Llength2 = (crpLx-x0)*(crpLx-x0)+(crpLy-y0)*(crpLy-y0)
    Rlength2 = (crpRx-x0)*(crpRx-x0)+(crpRy-y0)*(crpRy-y0)

    # あまりに大きな値は異常　 正規化の設定上、半径が150を超えることはない
    if Llength2 > 22500:
        if debugmode:
            print("tooLongErr")
        crpRx, crpRy = np.inf, np.inf
        Llength2 = np.inf
    if Rlength2 > 22500:
        if debugmode:
            print("tooLongErr")
        crpLx, crpLy = np.inf, np.inf
        Rlength2 = np.inf

    # 左右のバランスから考えた異常値
    if Rlength2 != np.inf and Llength2/Rlength2 < 0.01:  # 左が右の1%に満たない長さである -> 片方だけ異常値として扱う
        if debugmode:
            print("L radius too short", a)
        crpLx, crpLy = np.inf, np.inf
    elif Llength2 != np.inf and Rlength2/Llength2 < 0.01:  # 右が左の1%に満たない長さである -> 異常値として扱う
        if debugmode:
            print("R radius too short", a)
        crpRx, crpry = np.inf, np.inf
    # 以上のチェックに引っかからなければそのまま登録する
    return [crpLx, crpLy], [crpRx, crpRy]

# (34) 1点を通る直線（x0,y0を通り傾きdy/dx)と輪郭画像の交点を求める


def crossPointsLRonImg0(img, x0, y0, dx, dy):
    # 輪郭線を描いた画像を用意する
    con = getContour(img)
    rdcimg = np.zeros_like(img)  # 描画キャンバスの準備
    cv2.drawContours(rdcimg, con, -1, 255, thickness=1)
    canvas1 = rdcimg.copy()/255  # 輪郭線を描いたキャンバス
    canvas2 = np.zeros_like(rdcimg)  # 白紙の描画キャンバス
    # (x0,y0) 中心軸上の点
    # dy/dx  その点での軸線の傾き
    if dx == 0:
        print("端点断面が垂直になっていますので処理を続けられません")
        sys.exit()

    acc = - dy/dx if dx != 0 else np.inf  # 法線の傾き
    # x0 から 左に -10*x0 離れた点と右に 10*x0 離れた点を結ぼうとしている
    (lx, rx) = (-4*x0, 6*x0) if dx != 0 else (x0, x0)
    ly = y0 - 5*x0/acc if dx != 0 else 0  # dx==0 の時は上ではねているが、将来のため
    ry = y0 + 5*x0/acc if dx != 0 else 320
    lx, rx = int(round(float(lx))), int(round(float(rx)))
    ly, ry = int(round(float(ly))), int(round(float(ry)))

    canvas2 = cv2.line(canvas2, (lx, ly), (rx, ry),
                       1, 2)  # 幅3（2*2-1）の直線を明るさ１で描く
    canvas = canvas1 + canvas2

    # 左の交点　　　重なった場所は値が２となっている.
    cross_pointsL = np.where(canvas[:, :int(x0)] == 2)
    # 右の交点　　　重なった場所は値が２となっている.
    cross_pointsR = np.where(canvas[:, int(x0):] == 2)

    if len(cross_pointsL[0]) == 0:
        crpLy, crpLx = np.inf, np.inf
    elif len(cross_pointsL[0]) == 1:  # 唯一なら決まり
        crpLy, crpLx = cross_pointsL[0][0], cross_pointsL[1][0]
    else:
        diff = np.array(
            [dx*dx+dy*dy for (dx, dy) in zip(cross_pointsL[1]-x0, cross_pointsL[0]-y0)])
        difforder = np.argsort(diff)
        min1x = cross_pointsL[1][difforder[0]]  # 最短
        min2x = cross_pointsL[1][difforder[1]]  # 　２位
        min1y = cross_pointsL[0][difforder[0]]  # 最短
        min2y = cross_pointsL[0][difforder[1]]  # 　２位
        if abs(min1x-min2x) > 1 or abs((min1y-min2y)) > 1:  # 最短と2位が近傍関係にないなら最短で決まり
            crpLy, crpLx = min1y, min1x
        else:  # 最短と２位が隣接点の場合、どちらが法線と直交かで決める
            dotp1 = (min1x-x0)*dx+(min1y-y0)*dy  # 内積
            dotp1 = abs(dotp1/np.sqrt(diff[difforder[0]]))
            dotp2 = (min2x-x0)*dx+(min2y-y0)*dy
            dotp2 = abs(dotp2/np.sqrt(diff[difforder[1]]))
            if dotp1 <= dotp2:  # 小さい方がより法線上にあると判断する
                # print("L1win2","x0y0",x0,y0,"min1",min1x,min1y,"min1",min2x,min2y)
                crpLy, crpLx = min1y, min1x
            else:
                #print("LOoops 2win1","x0y0",x0,y0,"min1",min1x,min1y,"min1",min2x,min2y)
                crpLy, crpLx = min2y, min2x

    if len(cross_pointsR[0]) == 0:
        crpRy, crpRx = np.inf, np.inf
    elif len(cross_pointsR[0]) == 1:  # 唯一なら決まり
        crpRy, crpRx = cross_pointsR[0][0], cross_pointsR[1][0]
    else:
        diff = np.array(
            [dx*dx+dy*dy for (dx, dy) in zip(cross_pointsR[1]-x0, cross_pointsR[0]-y0)])
        difforder = np.argsort(diff)
        min1x = cross_pointsR[1][difforder[0]]  # 最短
        min2x = cross_pointsR[1][difforder[1]]  # 　２位
        min1y = cross_pointsR[0][difforder[0]]  # 最短
        min2y = cross_pointsR[0][difforder[1]]  # 　２位
        if abs(min1x-min2x) > 1 or abs((min1y-min2y)) > 1:  # 最短と2位が近傍関係にないなら最短で決まり
            crpRy, crpRx = min1y, min1x
        else:  # 最短と２位が隣接点の場合、どちらが法線と直交かで決める
            dotp1 = (min1x-x0)*dx+(min1y-y0)*dy  # 内積
            dotp1 = abs(dotp1/np.sqrt(diff[difforder[0]]))
            dotp2 = (min2x-x0)*dx+(min2y-y0)*dy
            dotp2 = abs(dotp2/np.sqrt(diff[difforder[1]]))
            if dotp1 <= dotp2:
                crpRy, crpRx = min1y, min1x
            else:
                crpRy, crpRx = min2y, min2x

    return (crpLx, crpLy), (crpRx+x0, crpRy)

# OverFitting判定　標本点間の異常判定
#実輪郭の各標本点間を4分割し、4分位点３つと標本点２つの5点を区間代表とし、近似曲線の対応区間で対応する５点との距＃
def isOverFitting(func,ts,cont,err_th=1.0,of_th=1.0):
    if len(cont) == 0:
        return []
    Nsamples = len(ts)
    # 実輪郭線側の標本点間弧長を計算する
    axlength = np.array(cv2.arcLength(cont,closed=False))  # 全周の長さ
    span = axlength/(Nsamples-1) # 平均標本点間距離
    lengths = np.array([cv2.arcLength(cont[:i+1], closed=False) for i in range(len(cont))])
                                                    # 始点から全輪郭点にいたる弧長
    spidx = np.array([np.abs(lengths - i).argmin() for i in np.linspace(0, axlength, Nsamples)])
                                                    # 等間隔にとった標本点のインデックス
    rs1 = []
    for i in range(Nsamples-1):
        qls = np.linspace(lengths[spidx[i]],lengths[spidx[i+1]],5)
        qidx = np.array([np.abs(lengths - l).argmin() for l in qls])
        rq5 = np.array([cont[s] for s in qidx]) 
        rs1.append(rq5) # 各区分の両端と4分割点計5点ずつのリスト
    # 近似曲線側の弧長を計算する
    rs2 = []
    fx,fy = func
    nfx, nfy = lambdify('t', fx, "numpy"), lambdify('t', fy, "numpy")
    for i in range(Nsamples-1):
        d5 = getDenseParameters(func, st=ts[i], et=ts[i+1], n_samples=5) # 標本点のパラメタ間を4分割
        aq5 = np.array([[nfx(s),nfy(s)] for s in d5]) # 近似曲線上で区間を4等分する座標のリスト
        rs2.append(aq5)
    # 代表5点の残差の標準偏差 
    difs = np.array([np.std(np.sum((rq5-aq5)*(rq5-aq5),axis=1)) for (rq5,aq5) in zip(rs1,rs2)])
    #q75, q25 = np.percentile(difs, [75,25]) # 四分位点
    #odds0 = np.where((difs>q75+1.5*(q75-q25))) # 異常値のインデックス
    odds = np.where(difs > err_th*of_th*span)[0] # 
    # print(odds,[difs[i] for i in odds0])
    return odds  

# (-1)変数データのストアとリストア
# 変数内データを pickle 形式で保存
def storePkl(val, fname, folder="."):
    os.makedirs(folder, exist_ok=True)
    f = open(folder+"/"+fname, 'wb')
    pickle.dump(val, f)
    f.close
