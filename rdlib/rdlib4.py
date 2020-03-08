from glob import glob
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

# from sympy import *
from sympy import diff,Symbol,Matrix,symbols,solve,simplify,binomial
from sympy.abc import a,b,c
# init_session()
from sympy import var
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

def assertglobal(params,verbose=False):
    global CONTOURS_APPROX, HARRIS_PARA, CONTOURS_APPROX, SHRINK, \
            HARRIS_PARA, GAUSSIAN_RATE1, GAUSSIAN_RATE2, UNIT, RPARA
    for item in params:
        if item == 'UNIT':
            UNIT = params[item] # 最終的に長い方の辺をこのサイズになるよう拡大縮小する
        elif item == 'SHRINK':
            SHRINK = params[item] # 収縮膨張で形状を整える時のパラメータ
        elif item == 'CONTOURS_APPROX':
            CONTOURS_APPROX = params[item] # 輪郭近似精度
        elif item == 'HARRIS_PARA':
            HARRIS_PARA = params[item] # ハリスコーナー検出で、コーナーとみなすコーナーらしさの指標  1.0 なら最大値のみ
        elif item == 'CONTOURS_APPROX' :
            CONTOURS_APPROX = params[item] # 輪郭近似精度
        elif item == 'GAUSSIAN_RATE1':
            GAUSSIAN_RATE1= params[item] # 先端位置を決める際に使うガウスぼかしの程度を決める係数
        elif item == 'GAUSSIAN_RATE2':
            GAUSSIAN_RATE2 = params[item] # 仕上げに形状を整えるためのガウスぼかしの程度を決める係数
        elif item == 'RPARA':
            RPARA = params[item]# 見込みでサーチ候補から外す割合
        # if verbose:
        #     print(item, "=", params[item])

assertglobal(params = {
    'HARRIS_PARA':1.0, # ハリスコーナー検出で、コーナーとみなすコーナーらしさの指標  1.0 なら最大値のみ
    'CONTOURS_APPROX':0.0002, # 輪郭近似精度
    'SHRINK':0.8, # 0.75 # 収縮膨張で形状を整える時のパラメータ
    'GAUSSIAN_RATE1':0.2, # 先端位置を決める際に使うガウスぼかしの程度を決める係数
    'GAUSSIAN_RATE2':0.1, # 仕上げに形状を整えるためのガウスぼかしの程度を決める係数
    'UNIT':256, # 最終的に長い方の辺をこのサイズになるよう拡大縮小する
    'RPARA':1.0 # 見込みサーチのサーチ幅全体に対する割合 ３０なら左に３０％右に３０％の幅を初期探索範囲とする
})


# OpenCV バージョン３とバージョン４の輪郭抽出関数の違いを吸収する関数
import cv2
def cv2findContours34(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE):
    v = list(cv2.__version__)
    if v[0] == '3':
        _img, cont, hier = cv2.findContours(image=image, mode=mode, method=method)
    else : # if v[0] == '4':å
        cont, hier = cv2.findContours(image=image.copy(), mode=mode, method=method)
    return cont, hier

# (1)画像のリストアップ
# 指定フォルダ内の画像の収集
# 画像パスの収集
def collectimagepaths(path, imgexts=['jpg','jpge','png']):
    allfiles = glob(path+'/*')
    imgfiles = []
    for x in allfiles:
        if os.path.splitext(x)[-1][1:] in imgexts:
            imgfiles.append(x)
    imgfiles.sort() # ファイル名の順に並び替える
    return imgfiles

# 画像の収集
def collectimages(path, imgexts=['jpg','jpge','png']):
    imgfiles = collectimagepaths(path, imgexts)
    imgs = [cv2.imread(afile,-1) for afile in imgfiles]
    return imgs

# サムネイルの作成
def makethumbnail(path, savedir='.',imgexts=['jpg','jpge','png']):
    imgfiles = collectimagepaths(path, imgexts)
    i = 0
    sam = Image.new('RGB', (500,100*((len(imgfiles)+4)//5)),(0,0,0))
    row = col = 0
    for rad in imgfiles:
        img = Image.open(rad, 'r')
        img.thumbnail((100, 100))
        sam.paste(img,(col,row))
        col += 100
        if col == 500:
            col = 0
            row += 100
    dirname = os.path.splitext(os.path.basename(path))[0]
    sam.save(savedir+os.sep+'{}THUM.PNG'.format(dirname), 'PNG')
    return sam

# (2)画像の表示
# プロット用関数
def plotimg(img,layout="111"):
    if img.ndim == 2:
        pltgry(img,layout)
    elif img.ndim ==3:
        pltcol(img,layout)

def pltgry(img,layout="111"):
    plt.subplot(layout)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))

def pltcol(img,layout="111"):
    plt.subplot(layout)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# (3) mkparaimage で２枚並べた画像を表示
def mkparaimage2(img1,img2):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    img11 = np.zeros((h1,w1,3),'uint8')
    if img1.ndim == 2:
        img11[:,:,0] = img11[:,:,1] = img11[:,:,2] = img1
    else:
        img11=img1
    if img2.ndim == 2:
        img22 = np.zeros((h2,w2,3),'uint8')
        img22[:,:,0]=img22[:,:,1]=img22[:,:,2]=img2
    else:
        img22=img2
    paraimg = 255*np.ones((max(h1,h2),w1+w2+10,3),dtype=np.uint8)
    
    paraimg[0:h1,0:w1,:] = img11
    paraimg[0:h2,w1+10:,:]=img22
    
    return paraimg

def mkparaimage(imglist):
    if len(imglist) == 0:
        return
    if len(imglist) == 1:
        return imglist[0]
    if len(imglist) == 2:
        return mkparaimage2(imglist[0],imglist[1])
    return mkparaimage2(imglist[0],mkparaimage(imglist[1:]))

# (3) マージンをつける
MinimamMargin = 10 
def makemargin(img,mr=1.5,mm = MinimamMargin):
    # 画像サイズが元の短径の mr 倍になるようにマージンをつける
    h,w = img.shape[:2]
    margin = int((mr-1)*min(h,w)//2 + 1) 
    margin = mm if margin < mm else margin
    w2 = w + 2*margin
    h2 = h + 2*margin
    x1,y1 = margin,margin
    if len(img.shape)==2:
        img2 = np.zeros((h2,w2),np.uint8)
    else:
        img2 = np.zeros((h2,w2,img.shape[2]),np.uint8)
    img2[y1:y1+h,x1:x1+w] = img
    return img2

# マージンのカット
# 白黒画像(0/255)が前提
def cutmargin(img,mr=1.0,mm=0,withRect=False):
    # default ではバウンディングボックスで切り出し
    # makemargin(切り出した画像,mr,mm)でマージンをつけた画像を返す
    # withRect = True の場合は切り出しの rect も返す
    if len(img.shape) > 2:
        gryimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gryimg = img.copy()
    bimg,data,ami = getMajorWhiteArea0(img=gryimg)
    x,y = data[ami][0:2]
    w,h = data[ami][2:4]
    bimg = bimg[y:y+h,x:x+w] # マージンなしで切り出して返す
    bimg = makemargin(bimg,mr=mr,mm=mm)
    if withRect:
        return bimg,x,y,w,h
    else:
        return bimg

# (4) 最大白領域の取り出し
def getMajorWhiteArea0(img, order=1):
    # order 何番目に大きい領域を取り出したいか
    # dilation 取り出す白領域をどれだけ多めにするか
    # binary 返す画像を２値化するかどうか
    if img.ndim == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # カラーの場合はグレイ化する
    _ret,bwimg = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ２値化
    # 背景が最大の領域であるという前提を用いているが、そうでない可能性があるのでいったん黒を付け足す
    limg = np.zeros_like(bwimg)
    bwimg = np.c_[bwimg,limg]
    _lnum, labelimg, cnt, _cog =cv2.connectedComponentsWithStats(bwimg) # ラベリング
    areaindexs = np.argsort(-cnt[:,4])
    if len(areaindexs) > order:
        areaindex = areaindexs[order] # order 番目に大きい白領域のインデックス
    else:
        areaindex = areaindexs[-1] # 指定した番号のインデックスが存在しないなら一番小さい領域番号
    labelimg[labelimg != areaindex] = 0
    labelimg[labelimg == areaindex] = 255
    w = labelimg.shape[1]//2
    labelimg = labelimg.astype(np.uint8)[:,:w]
    return labelimg, cnt, areaindex

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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
        fatlabelimg = cv2.dilate(limg,kernel,iterations = dilation) # 太らせる
        oimg[fatlabelimg == 0 ] = 0
    else:
        oimg[limg == 0 ] = 0
    if binary:
        oimg[limg>0] = 255
    return oimg

# (5) 処理結果画像（fimg)に処理前画像（bimg)の輪郭を描く
def draw2(bimg,fimg,thickness=2,color=(255,0,200)):
    bimg2 = getMajorWhiteArea(bimg,binary=True)
    if len(fimg.shape)>2:
        fimg2 = fimg.copy()
    else:
        fimg2 = cv2.cvtColor(fimg,cv2.COLOR_GRAY2BGR)
    # 処理前画像の輪郭
    # canvas = np.zeros_like(fimg2)
    canvas = fimg2.copy()
    _ret,bwimg = cv2.threshold(bimg2,128,255,cv2.THRESH_BINARY) # 白画素は255にする
    cnt,_hierarchy = cv2findContours34(bwimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = cv2.drawContours(canvas, cnt, -1, color, thickness)
    return canvas # 

# (6) (x1,y1)から（x2,y2) に向かう直線のX軸に対する角度
def getDegreeOfALine(x1,y1,x2,y2):
        dx = x2-x1
        dy = y2-y1
        if dx == 0 :
            if dy > 0:
                deg = 90
            else:
                deg = -90
        elif dx >0:
            deg = 180.0*np.arctan(dy/dx)/np.pi
        else:
            deg = 180*(1+np.arctan(dy/dx)/np.pi)
        return deg

# (7) (x1,y1)から（x2,y2) に向かう直線の延長上の十分離れた２点の座標
def getTerminalPsOnLine(x1,y1,x2,y2):
        dx = x2-x1
        dy = y2-y1
        s1 = int(x1 + 1000*dx)
        t1 = int(y1 + 1000*dy)
        s2 = int(x1 - 1000*dx)
        t2 = int(y1 - 1000*dy)
        return s1,t1,s2,t2

# (8) ガウスぼかし、膨張収縮、輪郭近似で形状を整える関数 RDreform()
# 形状の細かな変化をガウスぼかし等でなくして大まかな形状にする関数

# ガウスぼかしの程度を決めるカーネルサイズの自動決定
def calcksize(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
    gray = cv2.GaussianBlur(img,(7,7),0) # とりあえず (7,7)でぼかして2値化
    _ret,bwimg = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ２値化
    _lnum, labelimg, cnt, _cog =cv2.connectedComponentsWithStats(bwimg) # ラベリング
    if(len(cnt)<2):
        ksize = 3
    else:
        areamax = np.argmax(cnt[1:,4])+1 # ０番を除く面積最大値のインデックス
        maxarea = np.max(cnt[1:,4])
        ksize = int(np.sqrt(maxarea)/60)*2+1
    return ksize

def RDreform(img,order=1,ksize=0,shrink=SHRINK):
    # ksize : ガウスぼかしの量、shrink 膨張収縮による平滑化のパラメータ
    # order : 取り出したい白領域の順位
    # shrink : 膨張収縮による平滑化のパラメータ　

    # ガウスぼかしを適用してシルエットを滑らかにする
    # ガウスぼかしのカーネルサイズの決定

    if img.ndim > 2:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if img.sum() == 0: # 白領域が存在しない
        return img
    if ksize == 0:  # ぼかしのサイズが指定されていないときは最大白領域の面積を基準に定める
        ksize = calcksize(img)
    img2 = cv2.GaussianBlur(img,(ksize,ksize),0) # ガウスぼかしを適用
    img2 = getMajorWhiteArea(img2,order=order,dilation=2) # 指定白領域を少しだけ大きめに取り出す

    _ret,img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ２値化

    return RDreform_D(img2,ksize=ksize,shrink=SHRINK)

def RDreform_D(img,ksize=5,shrink=SHRINK):

    # 収縮・膨張によりヒゲ根を除去する
    area0 = np.sum(img) # img2 の画素数*255 になるはず
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize)) # 円形カーネル

    # ミディアンフィルタで小さなノイズを除去
    img = cv2.medianBlur(img, ksize)

    # 残った穴を埋める
    h,w = img.shape[:2]
    inv = 255-img
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    _retval, holes, _mask, _rect = cv2.floodFill(inv, mask, (0,0), newVal=0) 
    img = img + holes
    
    tmpimg = cv2.erode(img,kernel,iterations = 1) # 収縮１回目
    area1 = np.sum(tmpimg) # 収縮したので area0 より少なくなる

    n = 1 # 収縮回数のカウンタ

    while area0 > area1 and area1  > shrink*area0: # 面積が SHRINK倍以下になるまで繰り返す
        tmpimg = cv2.erode(tmpimg,kernel,iterations = 1)
        area1 = np.sum(tmpimg) 
        n += 1
    img3 = cv2.dilate(tmpimg,kernel,iterations = n) # 同じ回数膨張させる
    # あらためて輪郭を求め直す

    cnt,_hierarchy = cv2findContours34(img3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #  あらためて輪郭を抽出
    outimg = np.zeros_like(img3)
    if len(cnt) > 0:
        perimeter = cv2.arcLength(cnt[0],True) # 周囲長
        epsilon = CONTOURS_APPROX*perimeter # 周囲長をもとに精度パラメータを決定
        # 概形抽出
        approx = cv2.approxPolyDP(cnt[0],epsilon,True)

        # 輪郭を描いて埋める   
        outimg = cv2.drawContours(outimg, [approx], 0, 255, thickness=-1) 
    else:
        outimg = np.ones_like(img3)*255
    return outimg

# (9) Grabcut による大根領域の抜き出し
# GrabCutのためのマスクを生成する
def mkGCmask(img, order=1):
    # カラー画像の場合はまずグレー画像に変換
    gray = RDreform(img,order=order,ksize=0,shrink=SHRINK)

    # 大きめのガウシアンフィルタでぼかした後に大津の方法で２階調化
    ksize = calcksize(gray) # RDForm で使う平滑化のカーネルサイズ
    bsize = ksize # 
    blur = cv2.GaussianBlur(gray,(bsize,bsize),0)  # ガウスぼかし                        
    coreimg = getMajorWhiteArea(blur) # ２値化して一番大きな白領域
    coreimg[coreimg !=0 ] = 255
    
    # 膨張処理で確実に背景である領域をマスク
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize)) # 円形カーネル
    mask1 = 255-cv2.dilate(coreimg,kernel,iterations = ksize)
    
    # 収縮処理で確実に内部である領域をマスク
    mask2 = cv2.erode(coreimg,kernel,iterations = ksize)

    return mask1,mask2

# (10) 大根部分だけセグメンテーションし、結果とマスクを返す
def getRadish(img,order=1,shrink=SHRINK):
    # 白領域の面積が order で指定した順位の領域を抜き出す

    mask1,mask2 = mkGCmask(img,order=order)

    # grabcut　用のマスクを用意 
    grabmask = np.ones(img.shape[:2],np.uint8)*2
    # 
    grabmask [mask1==255]=0     # 黒　　背景　　
    grabmask [mask2==255]=1    # 白　前景
    
    # grabcut の作業用エリアの確保
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # MASK による grabcut 実行
    grabmask, bgdModel, fgdModel = cv2.grabCut(img,grabmask,None,bgdModel,fgdModel,20,cv2.GC_INIT_WITH_MASK)
    grabmask = np.where((grabmask==2)|(grabmask==0),0,1).astype('uint8')
    grabimg = img*grabmask[:,:,np.newaxis]
    silimg = np.zeros(grabimg.shape[:2],np.uint8)
    graygrabimg = cv2.cvtColor(grabimg,cv2.COLOR_BGR2GRAY)
    silimg[graygrabimg != 0] = 255 

    silimg = getMajorWhiteArea(silimg, order=1, dilation=0)
    silimg = RDreform_D(silimg,ksize=calcksize(silimg),shrink=shrink)

    return grabimg, silimg

# (11) 重心の位置を求める
def getCoG(img):
    _lnum, _img, cnt, cog = cv2.connectedComponentsWithStats(img)
    areamax = np.argmax(cnt[1:,4])+1 # ０番を除く面積最大値のインデックス
    c_x,c_y = np.round(cog[areamax]) # 重心の位置を丸めて答える
    # x,y,w,h,areas = cnt[areamax] # 囲む矩形の x0,y0,w,h,面積
    return c_x,c_y,cnt[areamax]

# (12) 回転した上でマージンをカットした画像を返す
def rotateAndCutMargin(img,deg,c_x,c_y): 
    # 非常に稀であるが、回転すると全体が描画領域外に出ることがあるので作業領域を広く確保
    # mat = cv2.getRotationMatrix2D((x0,y0), deg-90, 1.0) # アフィン変換マトリクス
    bigimg = makemargin(img,mr=10) # 作業用のマージンを確保
    h3,w3 = img.shape[:2]
    h4,w4 = bigimg.shape[:2]
    
    if deg != 0:
        mat = cv2.getRotationMatrix2D((c_x+(w4-w3)/2,c_y+(h4-h3)/2), deg, 1.0) # アフィン変換マトリクス
        # アフィン変換の適用
        bigimg = cv2.warpAffine(bigimg, mat, (0,0),1)

    # 再び最小矩形を求めて切り出す。
    _nLabels, _labelImages, data, _center = cv2.connectedComponentsWithStats(bigimg) 
    ami = np.argmax(data[1:,4])+1 # もっとも面積の大きい連結成分のラベル番号　（１のはずだが念の為）
    resultimg = bigimg[data[ami][1]:data[ami][1]+data[ami][3],data[ami][0]:data[ami][0]+data[ami][2]]

    return resultimg

# (13) 大きさを正規化したシルエットの生成
def getNormSil(img,tiltzero=True,mr=1.5,unitSize=UNIT):
    #  img 入力画像
    #  tiltzero  True なら傾き補正する
    #  mr マージンなし画像に対するマージンあり画像のサイズ比率
    #  unitSize バウンダリの長辺をこの値にする
    
    img = img.copy()
    
    if tiltzero: # 傾き補正する
        img = tiltZeroImg(img)
        
    return makeUnitImage(img,mr=mr,unitSize=UNIT)
    
# (14) 長辺の mr 倍サイズの枠の中央に対象を配置した画像を返す
def makeUnitImage(img,mr=1.5,unitSize=UNIT):
    # 長辺が UNIT ピクセルになるよう縮小し、(mrxUNIT)x(mrxUNIT)の画像の中央に配置する。
    h,w = img.shape[:2]
    s_r = unitSize/w if w > h else unitSize/h #  縮小率    
    rsh,rsw = int(s_r*h),int(s_r*w) # リサイズ後のサイズ
    x0 = int((mr*unitSize-rsw)/2) # はめ込みの基準点
    y0 = int((mr*unitSize-rsh)/2)
    canvas = np.zeros((int(mr*unitSize),int(mr*unitSize)),np.uint8) # キャンバスの確保
    canvas[y0:y0+rsh,x0:x0+rsw] = cv2.resize(img,(rsw,rsh)) # リサイズして中央にはめ込み
    return canvas
    
# (15) 近似楕円の軸方向が水平垂直となるように回転補正した画像を求める
def tiltZeroImg(img):
    h,w = img.shape[:2]
    img0,rx,ry,rw,rh = cutmargin(img,mm=0,withRect=True) # バウンダリボックスで切り出し
    cnt = cv2findContours34(img0, 1, 2)[0][0] # 輪郭線抽出
    (gx0,gy0),(sr,lr),ang = cv2.fitEllipse(cnt) # 楕円近似
    #  fitEllipse の返す角度は長軸を垂直に立てるための角度である。
    #  もしも幅が高さより大きい個体であれば、回転角を９０度減ずる
    ang = ang-90 if rw > rh else ang

    # 重心の偏りぐあいを記録
    ghight0  = abs(rh/2 - gy0) # シルエット重心と矩形中心のy方向距離
    gright0  = abs(rw/2 - gx0) # シルエット重心と矩形中心のx方向距離

    # 近似楕円の軸が水平垂直となるよう重心周りに回転を加えた図形を求める
    img1 = rotateAndCutMargin(img0,ang,gx0,gy0)

    # あらためて重心位置を求める
    gx1,gy1,(_,_,w1,h1,_) = getCoG(img1)

    # 180度回転してしまう場合があるので、重心の偏りぐあいで判断し、必要なら回転
    if ghight0 > gright0 : # 重心の上下の偏りがと左右の偏りより大きい場合
        if (rh/2 > gy0) != (h1/2 > gy1): # 上下の
            img1 = np.rot90(np.rot90(img1))
    else:
        if (rw/2 > gx0) != (w1/2 > gx1):
            img1 = np.rot90(np.rot90(img1)) 
            
    return img1

# (16) 輪郭点列を得る
def getContour(img):
    # 輪郭情報 主白連結成分の輪郭点列のみ返す関数
    contours, _hierarchy = cv2findContours34(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # 輪郭線追跡
    cnt00 = contours[np.argmax([len(c) for c in contours])] # 最も長い輪郭
    return cnt00

# (17) 輪郭表現の相互変換
# 輪郭構造体をただのリストに変換  -> [[1,2],[2,3]....]
def contolist(con):
    return con.squeeze().tolist()

# リストを輪郭線構造体に変換  -> array([[[1,2]],[[2,3]],...])
def listtocon(list):
    return np.array([[p] for p in list])

# (18)  輪郭の描画
def drawContours(canvas,con,color=255,thickness=1):
    if type(con) == np.ndarray:
        if con.ndim == 3: # 普通の輪郭情報
            cv2.drawContours(canvas,con,-1, color=color, thickness=thickness)
        elif con.ndim == 2: # 片側のみの輪郭
            cv2.polylines(canvas,[con],isClosed=False, color=color, thickness=thickness)
    elif type(con) == list:
        for c in con:
            drawContours(canvas,c,color=255,thickness=1)

# (19) 中心軸端点の推定
from statistics import mean
def findTips(img,con=[],top=0.1,bottom=0.9,topCD=1.0, bottomCD=0.8):
    # 入力　
    #   img シルエット画像
    #   con 輪郭点列　（なければ画像から作る）
    # パラメータ
    #   top 中心軸上端点の候補探索範囲　高さに対する割合
    #   bottom 中心軸下端点の候補探索範囲　
    #   topCD  中心軸上端点らしさの評価データを収集する範囲
    #   bottomCD 中心軸下端点らしさの評価データを収集する範囲
    # 出力
    #   con  輪郭点列
    #   topTip  中心軸上端点の輪郭番号
    #   bottomTip 中心軸下端点の輪郭番号
    #   symtops 各輪郭点の中心軸上端らしさの評価データ
    #   symbottoms 各輪郭点の中心軸下端らしさの評価データ
    
    if len(con)==0: # 点列がすでにあるなら時間短縮のために与えてもよい
        con = getContour(img)
    conlist = con.squeeze().tolist() # 輪郭点列のリスト
    gx,gy,(x0,y0,w,h,a) = getCoG(img) # 重心とバウンディングボックス
    N = len(conlist) # 輪郭点列の数)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # オープニング（収縮→膨張）平滑化した図形を求める
    img1 = cv2.morphologyEx(img.copy(), cv2.MORPH_OPEN, kernel,iterations = 5)
    HL = (N+len(getContour(img1)))//4 #  平滑化図形の輪郭長
    
    def calcval(i,irrend):
        data = []
        vc = 0
        for n in range(1,irrend): # 距離が近すぎると誤差が大きいので３から
            p0,p1,p2 = conlist[i],conlist[(i-n)%N],conlist[(i+n)%N]
            v1 = (p2[0]-p1[0],p2[1]-p1[1]) # p1p2 ベクトル
            p3 = ((p1[0]+p2[0])/2,(p1[1]+p2[1])/2) # p3 = p1とp2の中点
            v2 = (p3[0]-p0[0],p3[1]-p0[1]) # p0p3 ベクトル
            nv1 = np.linalg.norm(v1) # p1p2 の長さ
            nv2 = np.linalg.norm(v2) # p0p3 の長さ
            if nv2 > 0:
                data.append(np.dot(v1,v2)/nv1/nv2)
        if len(data) == 0:
            return -1
        else:
            return abs(mean(data))
    
    # 上部の端点の探索
    symtops = []
    for i in range(N):
        if conlist[i][1] - y0 >= top*h: # バウンディングボックス上端からの距離
            val = -1
        else:
            val = calcval(i,irrend=int(topCD*HL))
        symtops.append(val)
    m = np.max(symtops) # 探索対象のうちの最大評価値
    for i in range(N):
        if symtops[i] < 0: 
            symtops[i] = m
    topTip = np.argmin(symtops)
    
    # 下部の端点の探索
    symbottoms = []
    for i in range(N):
        if conlist[i][1] - y0 < bottom*h: # バウンディングボックス上端からの距離
            val = -1
        else:
            val = calcval(i,irrend=int(bottomCD*HL))
        symbottoms.append(val)
    m = np.max(symbottoms) # 探索対象のうちの最大評価値
    for i in range(N):
        if symbottoms[i] < 0: 
            symbottoms[i] = m
    bottomTip = np.argmin(symbottoms)
    return con,topTip,bottomTip,symtops,symbottoms


## (20) 上端・末端情報に基づき輪郭線を左右に分割する
def getCntPairWithCntImg(rdcimg,dtopx,dtopy,dbtmx,dbtmy,dtopdr=10,dbtmdr=10):
    # drcimg: ダイコンの輪郭画像
    # (dtopx,dtopy) dtopdr　上部削除円中心と半径
    # (dbtmx,dbtmy) dbtmdr 　下部削除円中心と半径
    
    # 中心軸上端部と末端部に黒で円を描いて輪郭を２つに分離
    canvas = rdcimg.copy()
    # まず上端を指定サイズの円で削る
    canvas = cv2.circle(canvas,(dtopx,dtopy),dtopdr,0,-1)  

    def bracket2to1(cnt):    
        cnt = [[x,y] for [[x,y]] in cnt]
        return cnt
        
    while True:
        # 次に末端を削る。末端は細いので、左右の輪郭が縮退している場合があり、削除円が小さいと輪郭が分離できず処理が進められない。
        canvas = cv2.circle(canvas,(dbtmx,dbtmy),dbtmdr,0,-1) 
    
        # 輪郭検出すれば２つの輪郭が見つかるはず。
        nLabels, _labelImages = cv2.connectedComponents(canvas)
        if nLabels >= 3: # 背景領域を含めると３以上の領域になっていれば正しい
            break
        dbtmdr = dbtmdr + 2 # ラベル数が　３（背景を含むので３） にならないとすれば先端が削り足りない可能性が最も高いので半径を増やしてリトライ   
        
    contours, hierarchy = cv2findContours34(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
    maxcnt_i = np.argmax(np.array([len(c) for c in contours]))
    # cnt0 = bracket2to1(contours[maxcnt_i]) # 最も長い輪郭
    cnt0 = contours[maxcnt_i].squeeze() # 最も長い輪郭
    contours = contours[:maxcnt_i]+contours[maxcnt_i+1:]
    maxcnt_i = np.argmax(np.array([len(c) for c in contours]))
    # cnt1 = bracket2to1(contours[maxcnt_i])
    cnt1 = contours[maxcnt_i].squeeze()
    
    # 分岐のない線図形の輪郭は、トレースが端点から始まれば１箇所、途中からなら２箇所折り返しがある。端点と折り返し、
    # もしくは、折り返しと折り返しの間を取り出すことで、重複のない輪郭データとする
    i1 = 0
    for i in range(int(len(cnt0))-1):
        if np.all(cnt0[i-1] == cnt0[i+1]): 
            i0,i1= i1,i
    cnt0 = cnt0[i0:i1+1]
    if cnt0[0][1] > cnt0[-1][1]: 
        cnt0 = cnt0[::-1]
    
    i1 = 0
    for i in range(int(len(cnt1))-1):
        if np.all(cnt1[i-1] == cnt1[i+1]):
            i0,i1= i1,i
    cnt1 = cnt1[i0:i1+1]
    if cnt1[0][1] > cnt1[-1][1]:
        cnt1 = cnt1[::-1]
    
    # 中程の点を比べて左にある方を左と判定する。
    c0 = cnt0[int(len(cnt0)/2)][0]
    c1 = cnt1[int(len(cnt1)/2)][0]
    if  c0 > c1: 
            cntl,cntr = cnt1,cnt0
    else:
            cntr,cntl = cnt1,cnt0

    return cntl, cntr

# (21) 与えられたダイコン画像の輪郭を左右に分割する
def getCntPairWithImg(rdimg,top=0.1,bottom=0.9,topCD=1.0, bottomCD=0.8,topdtopdr=10,dbtmdr=10):
    # drimg: ダイコンの画像
    # top,bottom,topCD,bottomCD ：findTips() に与えるパラメータ
    # dtopdr,dbtmdr:　getCntPairWithCntImg() に与えるパラメータ
    con,topTip,bottomTip,symtops,symbottoms = findTips(rdimg,top=top,bottom=bottom,topCD=topCD, bottomCD=bottomCD)
    conlist = contolist(con)
    dtopx,dtopy = conlist[topTip]
    dbtmx,dbtmy = conlist[bottomTip]
    rdcimg = np.zeros_like(rdimg)  # 描画キャンバスの準備
    cv2.drawContours(rdcimg,con, -1, 255,thickness=1)
    conLeft,conRight = getCntPairWithCntImg(rdcimg,dtopx,dtopy,dbtmx,dbtmy,dtopdr=10,dbtmdr=10)
    return conLeft,conRight

#　(22) 座標リストから等間隔で指定した数の標本を抜き出す。
def getSamples(cont,N=20,mode='Equidistant'):
    if mode=='Equidistant':
        axlength = np.array(cv2.arcLength(cont, closed=False)) # 弧長
        lengths = np.array([cv2.arcLength(cont[:i+1],closed=False)  for i in range(len(cont))]) # 各点までの弧長の配列
        return np.array([cont[np.abs(np.asarray(lengths) - i).argmin()] for i in np.linspace(0,axlength,N)])
    else: # 'Simple' 実際にはなんでもOK 
        return cont[list(map(int,np.linspace(0, len(cont)-1,N)))]

#  (23) N次ベジエフィッティング
class BezierCurve:
    
    # samples:標本点の座標の配列 （ opencv 輪郭形式、numpy 配列 ）
    # initialpara:初期の仮パラメータ値
    # N:ベジエ曲線の次数, precPara:近似精度（繰返し計算の打ち切りしきい値）
    # order 次数、openmode: 両端点フリー、Falseの時は両端点固定
    
    precPara = 0.1 # 繰り返しにおけるパラメータ変動幅の平均値に対するしきい値
    errorThres = 0.5 # 繰り返し打ち切りを決める、あてはめ誤差の平均値に対するしきい値
    paraResolution = 0.001 # サンプル点の最寄り点探索の解像度
    
    debugmode=False

    # インスタンスの初期化
    def __init__(self, samples, N=5, initialPara=[], openmode=False):
        self.N = N
        self.samples = samples # 標本点のリスト（opencv の輪郭点表現形式準拠、numpy 配列）
        self.openmode = openmode
        
        # もし、inf データが含まれるならば、補間する（計算で求められた座標データがサンプルの場合にありうる）
        samples = self.interporation(samples) 
        # パラメータの初期化
        if len(initialPara) > 0:  # 引数として与えられているならそれを使う
            if len(initialPara) != len(samples): # サンプル数と一致しない
                print("DATA NUMBER MISMATCH")
                sys.exit()
            else:
                self.tpara = initialPara
        else: # 初0～1をリニアに刻む
            axlength = np.array(cv2.arcLength(samples, False)) # 点列に沿って測った総経路長
            # 各サンプル点の始点からの経路長の全長に対する比を、各点のベジエパラメータの初期化とする
            self.tpara = [cv2.arcLength(samples[:i+1],False)  for i in range(len(samples))]/axlength
            
        # ベジエ曲線を定義するのに使うシンボルの宣言
        P = [Symbol('P' + str(i)) for i in range(N+1)] # 制御点を表すシンボル（数値変数ではない）
        px = [var('px'+str(i)) for i in range(N+1)] # 制御点のx座標を表すシンボル
        py = [var('py'+str(i)) for i in range(N+1)] # 制御点のy座標を表すシンボル
        t= symbols('t')
        v = var('v')
        for i in range(N+1):     # 制御点のシンボルと成分の対応付け
            P[i] = Matrix([px[i],py[i]]) 
            
        # N次のベジエ曲線の定義式制御点 P0～PN とパラメータ　　t　の関数として定義
        v = 1-t
        bezN = Matrix([0,0])
        for i in range(0,N+1):
            bezN = bezN + binomial(N,i)*v**(N-i)*t**i*P[i]
        self.bezN = bezN
            
    # 解なしの部分に np.inf が入っているのでその抜けを前後から推定してデータを埋める
    def interporation(self,plist):
        # plist : np.inf が混入している可能性のある座標の numpy array
        while np.sum(plist) == np.inf: # np.inf を含むなら除去を繰り返す
            for i in range(len(plist)):
                if np.sum(plist[i]) == np.inf :
                    print("欠",end="")
                    if (i !=0 and i !=len(plist)-1) and np.sum(plist[i-1]+plist[i+1]) != np.inf: # 当該は無限で、前後は無限ではない場合
                        plist = np.r_[plist[0:i],[(plist[i-1]+plist[i+1])/2],plist[i+1:]]
                    elif len(plist[i:])>=3 and np.sum(plist[i+1]+plist[i+2]) != np.inf:
                        plist = np.r_[plist[0:i],[plist[i+2]-2*(plist[i+2]-plist[i+1])],plist[i+1:]]
                    elif len(plist[0:i])>=2 and np.sum(plist[i-1]+plist[i-2]) != np.inf:
                        plist = np.r_[plist[0:i],[plist[i-2]-2*(plist[i-2]-plist[i-1])],plist[i+1:]]
        print("")
        return plist
    
    # 曲線 linefunc(t) 上で座標(x,y) に最も近い点のパラメータを2分サーチして探す関数
    def nearest(self,x,y,linefunc,pmin,pmax):
        t= symbols('t')
        (funcX,funcY) = linefunc # funcX,funcY は t の関数
        
        mid = (pmin+pmax)/2
        p = np.array([x,y])

        def us(p):
            x,y=p
            return np.array([float(x),float(y)])

        ps = funcX.subs(t,pmin),funcY.subs(t,pmin) # パラメータ最小点
        pm = funcX.subs(t,mid),funcY.subs(t,mid)  # 中間パラメータ点
        pe = funcX.subs(t,pmax),funcY.subs(t,pmax)  # パラメータ最大点
        ls = np.linalg.norm(us(ps) - p)
        lm = np.linalg.norm(us(pm) - p)
        le = np.linalg.norm(us(pe) - p)
        if pmax-pmin < BezierCurve.paraResolution : # 探索範囲がこのパラメータより狭くなれば探索終了
            m = min([ls,lm,le])
            if m == ls:
                return pmin
            elif m == le:
                return pmax
            else:
                return mid
        else:
            if ls < le:
                return self.nearest(x,y,linefunc,pmin,mid)
            else:
                return self.nearest(x,y,linefunc,mid,pmax)    

    #  パラメトリック曲線　linefunc 上で各サンプル点に最寄りの点のパラメータを対応づける
    def refineTparaN(self,linefunc,stt,end,pmin,pmax):
        if stt == end:
            return
        samples = self.samples
        nmid = (stt+end)//2  # 探索対象の中央のデータを抜き出す
        px,py = self.samples[nmid] # 中央のデータの座標
        midpara = self.nearest(px,py,linefunc,pmin,pmax)
        self.tpara[nmid] = midpara
        self.refineTparaN(linefunc,stt,nmid,pmin,midpara-BezierCurve.paraResolution)
        self.refineTparaN(linefunc,nmid+1,end,midpara+BezierCurve.paraResolution,pmax)

    # 曲線あてはめの実行
    def fit(self):

        samples = self.samples
        openmode = self.openmode
        N = self.N
        bezN = self.bezN

        # ベジエ曲線を定義するのに使うシンボルの宣言
        s,t= symbols('s,t')
        dx_ = [var('dx_'+str(i)) for i in range(N+1)] # 制御点におけるx微係数を表すシンボル
        dy_ = [var('dy_'+str(i)) for i in range(N+1)] # 制御点におけるy微係数を表すシンボル
        P = [Symbol('P' + str(i)) for i in range(N+1)] # 制御点を表すシンボル（数値変数ではない）
        px = [var('px'+str(i)) for i in range(N+1)] # 制御点のx座標を表すシンボル
        py = [var('py'+str(i)) for i in range(N+1)] # 制御点のy座標を表すシンボル
        
        # 最小自乗法の目的関数の一般式
        # 自乗誤差の一般式
        loss1 = (s - t)**2        
        def lossfunc(listA,listB):
            return sum([loss1.subs([(s,a),(t,b)]) for (a,b) in zip(listA,listB)])/2
        
        trynum = 0
        while True:
        
            trynum += 1
            linesamples = [bezN.subs(t,t_) for t_ in self.tpara] # 曲線上の点列の式表現
            linesamplesX = [x  for  [x,y] in linesamples] # X座標のリスト、式表現
            linesamplesY = [y  for  [x,y] in linesamples] # Y座標のリスト、式表現
            EsumX = lossfunc(listA=samples[:,0],listB=linesamplesX) #  X方向のずれの評価値
            EsumY = lossfunc(listA=samples[:,1],listB=linesamplesY) #  Y 方向のずれの評価値
            # px0,px1, px2, px3, ... py1, py2,py3 ...で偏微分

            if  not openmode : # 両端点を固定
                EsumX = EsumX.subs(px[-1],samples[-1][0])
                EsumY = EsumY.subs(py[-1],samples[-1][1])
                EsumX = EsumX.subs(px[0],samples[0][0])
                EsumY = EsumY.subs(py[0],samples[0][1])
            for i in range(0,N+1):
                dx_[i] = diff(EsumX,px[i])
                dy_[i] = diff(EsumY,py[i])  

            # 連立させて解く
            if not openmode :
                resultX = solve([dx_[i] for i in range(1,N)],[px[i] for i in range(1,N)])
                resultY = solve([dy_[i] for i in range(1,N)],[py[i] for i in range(1,N)])
            else : 
                resultX = solve([dx_[i] for i in range(N+1)],[px[i] for i in range(N+1)])
                resultY = solve([dy_[i] for i in range(N+1)],[py[i] for i in range(N+1)])

            if len(resultX) == 0 or len(resultY) == 0: # 方程式が解けない　非常にまれなケース
                return False,np.array([]),np.array([]),None,None,None

            # 解をベジエの式に代入
            if not openmode:
                bezresX = bezN[0].subs([(px[0],samples[0][0]),(px[-1],samples[-1][0])])
                bezresY = bezN[1].subs([(py[0],samples[0][1]),(py[-1],samples[-1][1])])
                for i in range(1,N):
                    bezresX = bezresX.subs(px[i],resultX[px[i]])
                    bezresY = bezresY.subs(py[i],resultY[py[i]])
            else: 
                bezresX = bezN[0]
                bezresY = bezN[1]           
                for i in range(0,N+1):
                    bezresX = bezresX.subs(px[i],resultX[px[i]])
                    bezresY = bezresY.subs(py[i],resultY[py[i]])

            rx,ry = resultX,resultY
            if not openmode:
                cpx = [samples[0][0]]+[rx[px[i]] for i in range(1,N)]+[samples[-1][0]]
                cpy = [samples[0][1]]+[ry[py[i]] for i in range(1,N)]+[samples[-1][1]]    
            else: # openmode
                cpx = [rx[px[i]] for i in range(N+1)]
                cpy = [ry[py[i]] for i in range(N+1)]

            tpara = self.tpara
            tpara0 = tpara.copy() # 前回の t の推定値
            
            self.refineTparaN((bezresX,bezresY),0,len(samples),0,1)
            diffpara, error = 0,0
            for i in range(len(tpara)) :
                diffpara += np.sqrt((tpara[i]-tpara0[i])**2) 
                sx,sy = samples[i]
                ox,oy = float(bezresX.subs(t,tpara[i])),float(bezresY.subs(t,tpara[i]))
                error += np.sqrt((sx-ox)**2+(sy-oy)**2)
            print(".",end='')
            diffpara = diffpara/len(tpara)*100 # パラメータの変化量の平均の100倍値
            error = error/len(tpara) # あてはめ誤差の平均
            if BezierCurve.debugmode: print("error:{}, drift:{}".format(error,diffpara))
            if trynum <= 5:
                if diffpara < BezierCurve.precPara and error < BezierCurve.errorThres:
                    break
            else:
                if diffpara < BezierCurve.precPara*1.05**(trynum-5): # 収束しない時のために、条件を徐々に緩めていく
                    break
        print("")
        return True,np.array(cpx),np.array(cpy),bezresX,bezresY,self.tpara
        # cpx,cpy 制御点、bezresX,bezresY ベジエ曲線の定義式
        # tpara 制御点   
        
    # デバッグモードのオンオフ
    def toggledebugmode():
        BezierCurve.debugmode = not BezierCurve.debugmode
        print("debugmode:",BezierCurve.debugmode)
        
    # パラメータのセットと表示　引数なしで呼ぶ出せば初期化
    def setParameters(reset=False,precPara=0.1,errorThres=0.5,paraResolution = 0.001):
        
        BezierCurve.precPara = precPara # 繰り返しにおけるパラメータ変動幅の平均値に対するしきい値
        BezierCurve.errorThres = errorThres # 繰り返し打ち切りを決める、あてはめ誤差の平均値に対するしきい値
        BezierCurve.paraResolution = paraResolution # サンプル点の最寄り点探索の解像度
        
        print("precPara:       ",precPara)
        print("errorThres:     ",errorThres)
        print("paraResolution: ",paraResolution)
        print("debugmode:      ",BezierCurve.debugmode)
        print("")
        
# 結果の描画
def drawBez(rdimg,bezL=None,bezR=None,bezC=None,cpl=None,cpr=None,cpc=None, 
             cntL=[],cntR=[],cntC=None, ladder=None,PosL=[],PosR=[],PosC=[],n_samples=20,saveImage=False,savepath=""):
    
    # いわゆる自乗誤差の一般式
    s,t= symbols('s,t')
    
    bezXl,bezYl = bezL if bezL != None else ([],[])
    bezXr,bezYr = bezR if bezR != None else ([],[])
    bezXc,bezYc = bezC if bezC != None else ([],[])
    cpxl,cpyl = cpl if cpl != None else ([],[])
    cpxr,cpyr = cpr if cpr != None else ([],[])
    cpxc,cpyc = cpc if cpc != None else ([],[])
    tplins50 = np.linspace(0, 1, 50)
    tplinsSP = np.linspace(0, 1, n_samples)
    
    plt.figure(figsize=(6,6),dpi=100)
    plt.gca().invert_yaxis() 
    plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を１：１に
    plt.imshow(192+(cv2.cvtColor(rdimg,cv2.COLOR_GRAY2RGB)/4).astype(np.uint8))
    # 左輪郭の描画
    if bezL != None:
        plotx = [bezXl.subs(t,tp) for tp in tplins50 ]
        ploty = [bezYl.subs(t,tp) for tp in tplins50 ]
        plt.plot(plotx,ploty,color = 'red')  
    if len(cntL) >0:
        plt.scatter(cntL[:,0],cntL[:,1],color ='blue',marker = '.') #  サンプル点
    if cpl != None:
        plt.scatter(cpxl,cpyl,color ='purple',marker = '*') #  制御点の描画
        for i in range(len(cpxl)) : plt.annotate(str(i),(cpxl[i],cpyl[i]))
    # 右輪郭の描画
    if bezR != None:
        plotx = [bezXr.subs(t,tp) for tp in tplins50 ]
        ploty = [bezYr.subs(t,tp) for tp in tplins50 ]
        plt.plot(plotx,ploty,color = 'red')  
    if len(cntR)  > 0:
        plt.scatter(cntR[:,0],cntR[:,1],color ='blue',marker = '.') #  サンプル点
    if cpr != None:
        plt.scatter(cpxr,cpyr,color ='red',marker = '*') #  制御点の描画
        for i in range(len(cpxr)):plt.annotate(str(i),(cpxr[i],cpyr[i]))
    # 中心軸の描画
    if bezC != None:
        plotx = [bezXc.subs(t,tp) for tp in tplins50 ]
        ploty = [bezYc.subs(t,tp) for tp in tplins50 ]
        plt.plot(plotx,ploty,color = 'red')  
        if cntC != None:
            plt.scatter(cntC[:,0],cntC[:,1],color ='blue',marker = '.') #  サンプル点
        if cpc != None:
            plt.scatter(cpxc,cpyc,color ='darkgreen',marker = '*') #  制御点の描画
            for i in range(len(cpxc)):plt.annotate(str(i),(cpxc[i],cpyc[i]))
                
        # ラダーの描画
        if  ladder== 'lr':  # 左右の同じパラメータ値の点を結ぶだけ
            plotSPlx = [bezXl.subs(t,tp) for tp in tplinsSP ]
            plotSPly = [bezYl.subs(t,tp) for tp in tplinsSP ]
            plotSPrx = [bezXr.subs(t,tp) for tp in tplinsSP ]
            plotSPry = [bezYr.subs(t,tp) for tp in tplinsSP ]       
            for x0,x1,y0,y1 in zip(plotSPlx,plotSPrx,plotSPly,plotSPry):
                plt.plot([x0,x1],[y0,y1],color = 'orange') 
                
        elif ladder == 'normal':
            # 中心軸上に設定したサンプル点における法線と両輪郭の交点のリストを求める。
            plot20lx = [xl if xl !=np.inf else np.inf for [xl,yl] in PosL ]
            plot20ly = [yl if yl !=np.inf else np.inf for [xl,yl] in PosL]
            plot20cx = PosC[:,0]
            plot20cy = PosC[:,1]
            plot20rx = [xr if xr !=np.inf else np.inf for [xr,yr] in PosR ]
            plot20ry = [yr if yr !=np.inf else np.inf for [xr,yr] in PosR ]
            for x0,x1,y0,y1 in zip(plot20lx,plot20cx,plot20ly,plot20cy):
                if x0 != np.inf and y0 !=np.inf:
                    plt.plot([x0,x1],[y0,y1],color = 'orange') 
            for x0,x1,y0,y1 in zip(plot20rx,plot20cx,plot20ry,plot20cy):
                if x0 != np.inf and y0 !=np.inf:
                    plt.plot([x0,x1],[y0,y1],color = 'orange') 
            if saveImage:
                pltsaveimage(savepath,'RAD')

# matplotlib で描いた画像の保存
def pltsaveimage(savepath,prefix):
        # 結果を保存する
        savedir,filename = os.path.split(savepath)
        #  _,subdir = os.path.split(savedir)
        os.makedirs(savedir, exist_ok=True) # 保存先フォルダがなければ作成
        savepath = os.path.join(savedir,prefix+filename)
        if os.path.exists(savepath):
            os.remove(savepath)
        print("TEST",savepath)
        plt.savefig(savepath)



# (-1)変数データのストアとリストア
import pickle
# 変数内データを pickle 形式で保存
def storePkl(val, fname, folder = "."):
    os.makedirs(folder, exist_ok=True)
    f = open(folder+"/"+fname,'wb')
    pickle.dump(val,f)
    f.close
    
# pickle 形式で保存されたデータを変数に復元
def loadPkl(fname, folder = "."):
    f = open(folder+"/"+fname,'rb')
    cat = pickle.load(f)
    f.close
    return cat
