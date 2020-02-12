from glob import glob
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

def assertglobal(params,verbose=False):
    global CONTOURS_APPROX, HARRIS_PARA, CONTOURS_APPROX, SHRINK, \
            HARRIS_PARA, GAUSSIAN_RATE1, GAUSSIAN_RATE2, UNIT, RPARA
    for item in params:
        if item == 'UNIT':
            UNIT = params[item] # 最終的に長い方の辺をこのサイズになるよう拡大縮小する
        elif item == 'SHRINK':
            SHRINK = params[item] # 0.75 # 収縮膨張で形状を整える時のパラメータ
        # elif item == 'CONTOURS_APPROX':
            CONTOURS_APPROX = params[item] # 輪郭近似精度
        # elif item == 'HARRIS_PARA':
            HARRIS_PARA = params[item] # ハリスコーナー検出で、コーナーとみなすコーナーらしさの指標  1.0 なら最大値のみ
        # elif item == 'CONTOURS_APPROX' :
            CONTOURS_APPROX = params[item] # 輪郭近似精度

        # elif item == 'GAUSSIAN_RATE1':
            GAUSSIAN_RATE1= params[item] # 先端位置を決める際に使うガウスぼかしの程度を決める係数
        # elif item == 'GAUSSIAN_RATE2':
            GAUSSIAN_RATE2 = params[item] # 仕上げに形状を整えるためのガウスぼかしの程度を決める係数
        # elif item == 'RPARA':
            RPARA = params[item]# 見込みでサーチ候補から外す割合
        # if verbose:
        #     print(item, "=", params[item])

assertglobal(params = {
    # 'HARRIS_PARA':1.0, # ハリスコーナー検出で、コーナーとみなすコーナーらしさの指標  1.0 なら最大値のみ
    # 'CONTOURS_APPROX':0.0002, # 輪郭近似精度
    'SHRINK':0.8, # 0.75 # 収縮膨張で形状を整える時のパラメータ
    # 'GAUSSIAN_RATE1':0.2, # 先端位置を決める際に使うガウスぼかしの程度を決める係数
    # 'GAUSSIAN_RATE2':0.1, # 仕上げに形状を整えるためのガウスぼかしの程度を決める係数
    'UNIT':256, # 最終的に長い方の辺をこのサイズになるよう拡大縮小する
    # 'RPARA':1.0 # 見込みサーチのサーチ幅全体に対する割合 ３０なら左に３０％右に３０％の幅を初期探索範囲とする
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
    if img1.ndim == 2:
        img11 = np.zeros((h1,w1,3))
        img11[:,:,0] = img11[:,:,1]=img11[:,:,2]=img1
    else:
        img11=img1
    if img2.ndim == 2:
        img22 = np.zeros((h2,w2,3))
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
def makemargin(img,mr=2):
    h,w = img.shape[:2]
    w2 = int(mr*w)
    h2 = int(mr*h)
    x1 = int((w2-w)/2)
    y1 = int((h2-h)/2)
    if len(img.shape)==2:
        img2 = np.zeros((h2,w2),np.uint8)
    else:
        img2 = np.zeros((h2,w2,img.shape[2]),np.uint8)
    img2[y1:y1+h,x1:x1+w] = img
    return img2

# (4) 指定した順位の面積の白領域の取り出し
def getMajorWhiteArea(img, order=1):
    # 白領域を面積順に並べたときの order 番目に大きな領域を取り出す
    if img.ndim == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # カラーの場合はグレイ化する
    _ret,bwimg = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ２値化
    _lnum, labelimg, cnt, _cog =cv2.connectedComponentsWithStats(bwimg) # ラベリング

    areaindex = np.argsort(-cnt[:,4])[order] # order 番目に大きい白領域のインデックス
    labelimg[labelimg != areaindex] = 0
    labelimg[labelimg == areaindex] = 255
    return labelimg.astype(np.uint8)

# (5) 処理結果画像（fimg)に処理前画像（bimg)の輪郭を描く
def draw2(bimg,fimg,thickness=2,color=(255,0,200)):
    bimg2 = getMajorWhiteArea(bimg)
    if len(fimg.shape)==3:
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

def calcksize(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
    gray = cv2.GaussianBlur(img,(7,7),0) # とりあえず (7,7)でぼかして2値化
    _ret,bwimg = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ２値化
    _lnum, labelimg, cnt, _cog =cv2.connectedComponentsWithStats(bwimg) # ラベリング
    areamax = np.argmax(cnt[1:,4])+1 # ０番を除く面積最大値のインデックス
    maxarea = np.max(cnt[1:,4])
    ksize = int(np.sqrt(maxarea)/60)*2+1
    return ksize

def RDreform(img,ksize=0,shrink=SHRINK,order=1):
    # ksize : ガウスぼかしの量、shrink 膨張収縮による平滑化のパラメータ
    # order : 取り出したい白領域の順位
    
    # 面積が指定された順位の白領域を取り出す
    img = getMajorWhiteArea(img, order
    
    # ガウスぼかしのカーネルサイズの決定
    if ksize == 0:  # ぼかしのサイズが指定されていないときは最大白領域の面積を基準に定める
        ksize = calcksize(img)

    # ガウスぼかしを適用してシルエットを滑らかにする
    img2 = cv2.GaussianBlur(img,(ksize,ksize),0) # ガウスぼかしを適用
    _ret,img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # ２値化
    
    # 収縮・膨張によりヒゲ根を除去する
    area0 = np.sum(img2) # img2 の画素数*255 になるはず
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize)) # 円形カーネル
    tmpimg = cv2.erode(img2,kernel,iterations = 1) # 収縮１回目
    area1 = np.sum(tmpimg) # 収縮したので area0 より少なくなる
    n = 1 # 収縮回数のカウンタ
    while area1  > shrink*area0: # 面積が SHRINK倍以下になるまで繰り返す
        tmpimg = cv2.erode(tmpimg,kernel,iterations = 1)
        area1 = np.sum(tmpimg) 
        n += 1
    img3 = cv2.dilate(tmpimg,kernel,iterations = n) # 同じ回数膨張させる
    # あらためて輪郭を求め直す
    # まず最大の領域を取り出す
    img3 = getMajorWhiteArea(img3,order)

    cnt,_hierarchy = cv2findContours34(img3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #  あらためて輪郭を抽出
    outimg = np.zeros_like(img3)
    perimeter = cv2.arcLength(cnt[0],True) # 周囲長
    epsilon = CONTOURS_APPROX*perimeter # 周囲長をもとに精度パラメータを決定
    # 概形抽出
    approx = cv2.approxPolyDP(cnt[0],epsilon,True)
    outimg = cv2.drawContours(outimg, [approx], 0, 255, thickness=-1) 
    
    return outimg

# (9) Grabcut による大根領域の抜き出し
def mkGCmask(img,thickness=5):
    # カラー画像の場合はまずグレー画像に変換
    if img.ndim == 3:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy() # 副作用はないと思うが、念のため

    # 大きめのガウシアンフィルタでぼかした後に大津の方法で２階調化
    ksize = calcksize(gray) # RDForm で使う平滑化のカーネルサイズ
    bsize = ksize # 
    blur = cv2.GaussianBlur(gray,(bsize,bsize),0)  # ガウスぼかし                        
    coreimg = getMajorWhiteArea(blur,order) # ２値化して一番大きな領域だけ抽出
    
    # 膨張処理で確実に背景である領域をマスク
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize)) # 円形カーネル
    mask1 = 255-cv2.dilate(coreimg,kernel,iterations = ksize+5)
    
    # 収縮処理で確実に内部である領域をマスク
    mask2 = cv2.erode(coreimg,kernel,iterations = ksize+5)

    return mask1,mask2

def getRadish(img,order=1):
    # 白領域の面積が order で指定した順位の領域を抜き出す

    mask1,mask2 = mkGCmask(img)

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

    return grabimg