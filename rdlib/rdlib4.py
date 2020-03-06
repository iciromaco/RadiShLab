from glob import glob
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

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
    'HARRIS_PARA':1.0, # ハリスコーナー検出で、コーナーとみなすコーナーらしさの指標  1.0 なら最大値のみ
    'CONTOURS_APPROX':0.0002, # 輪郭近似精度
    'SHRINK':0.8, # 0.75 # 収縮膨張で形状を整える時のパラメータ
    'GAUSSIAN_RATE1':0.2, # 先端位置を決める際に使うガウスぼかしの程度を決める係数
    'GAUSSIAN_RATE2':0.1, # 仕上げに形状を整えるためのガウスぼかしの程度を決める係数
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
    # withxy = True の場合は切り出しの rect も返す
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
    _lnum, labelimg, cnt, _cog =cv2.connectedComponentsWithStats(bwimg) # ラベリング
    areaindexs = np.argsort(-cnt[:,4])
    if len(areaindexs) > order:
        areaindex = areaindexs[order] # order 番目に大きい白領域のインデックス
    else:
        areaindex = areaindexs[-1] # 指定した番号のインデックスが存在しないなら一番小さい領域番号
    labelimg[labelimg != areaindex] = 0
    labelimg[labelimg == areaindex] = 255
    labelimg = labelimg.astype(np.uint8)
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

# 大根部分だけセグメンテーションし、結果とマスクを返す
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

# 重心の位置を求める
def getCoG(img):
    _lnum, _img, cnt, cog = cv2.connectedComponentsWithStats(img)
    areamax = np.argmax(cnt[1:,4])+1 # ０番を除く面積最大値のインデックス
    c_x,c_y = np.round(cog[areamax]) # 重心の位置を丸めて答える
    # x,y,w,h,areas = cnt[areamax] # 囲む矩形の x0,y0,w,h,面積
    return c_x,c_y,cnt[areamax]

# (9)重心と先端の位置を返す関数
#   先端位置はシルエットをガウスぼかしで滑らかにした上で曲率の高い場所
def getCoGandTip(src, showResult=False, useOldImage=True):    
    # useOldImage = True なら元の画像を使って結果を表示、Falseなら滑らかにした画像
    img = makemargin(src) # 作業用のマージンを確保
    img2 = img.copy() # 加工前の状態を保存
    # （あとでぼかすが、ぼかす前の）元画像の最大白領域の面積とバウンディングボックスを求める
    c_x,c_y,(_x0,y0,w,h,areas) = getCoG(img)
    print("1",_x0,y0,w,h,areas)
    radishwidth = areas/np.sqrt(w*w+h*h) # 面積をバウンディングボックスの対角の長さで割ることで大根の幅を大まかに見積もる
    # ガウスぼかしを適用してシルエットを滑らかにする
    ksize = int(GAUSSIAN_RATE1*radishwidth)*2+1 # ぼかし量  元の図形の幅に応じて決める
    img = cv2.GaussianBlur(img,(ksize,ksize),0) # ガウスぼかしを適用
    # ２値化してシルエットを求め直す
    _ret,img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # ２値化
    
    # コア全体の重心の位置を求める
    c_x,c_y,(_x0,y0,_w,h,areas) = getCoG(img)
    print("1",_x0,y0,w,h,areas)
    # 全体を囲む矩形の中間の高さ
    h2 = int(h/2)
    # Harris コーナ検出
    himg = np.float32(img)
    himg = cv2.cornerHarris(himg,blockSize=3,ksize=3,k=0.04)
    # コーナー度合いが最大の領域を求める
    wimg = np.zeros_like(img)
    wimg[himg>=HARRIS_PARA*himg[y0+h2:,:].max()]=255 # 下半分のコーナー度最大値の領域を２５５で塗りつぶす。
    # 最大値に等しい値の領域が１点とは限らないし、いくつかの点の塊になるかもしれない
    _lnum, _img, cnt, cog = cv2.connectedComponentsWithStats(wimg[y0+h2:,:])
    areamax = np.argmax(cnt[1:,4])+1 # ０番を除く面積最大値のインデックス
    t_x,t_y = np.round(cog[areamax]) # 重心の位置
    t_y += y0+h2

    # コーナーの場所のマーキング（デバッグ用）
    # himg = cv2.dilate(himg,None,iterations = 3)
    # img3[himg>=HARRIS_PARA*himg.max()]=[0,0,255]

    if showResult: # 
        if useOldImage:
            img3 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        else:
            img3 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        plt.figure(figsize=(10,7),dpi=75)
        img3=cv2.circle(img3,(int(t_x),int(t_y)),5,(0,255,0),2)
        img3=cv2.circle(img3,(int(c_x),int(c_y)),5,(255,255,0),2)
        x1,y1,x2,y2= getTerminalPsOnLine(c_x,c_y,t_x,t_y)
        img3=cv2.line(img3,(x1,y1),(x2,y2),(255,0,255),2)                 
        img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
        plt.subplot(122), plt.imshow(img3)
        plt.show()
        
    # 結果を返す (c_x,c_y) 重心　　(t_x,t_y)  先端の位置 img2 滑らかになったシルエット画像
    dx = int(img.shape[1]-src.shape[1])/2
    dy = int(img.shape[0]-src.shape[0])/2
    c_x,c_y,t_x,t_y = c_x-dx, c_y-dy, t_x-dx, t_y-dy 
    return c_x,c_y,t_x,t_y

# (10) 回転した上でマージンをカットした画像を返す
def roteteAndCutMargin(img,deg,c_x,c_y): 
    # 非常に稀であるが、回転すると全体が描画領域外に出ることがあるので作業領域を広く確保
    # mat = cv2.getRotationMatrix2D((x0,y0), deg-90, 1.0) # アフィン変換マトリクス
    bigimg = makemargin(img,mr=10) # 作業用のマージンを確保
    h3,w3 = img.shape[:2]
    h4,w4 = bigimg.shape[:2]
    
    if deg != 0:
        mat = cv2.getRotationMatrix2D((c_x+(w4-w3)/2,c_y+(h4-h3)/2), deg, 1.0) # アフィン変換マトリクス
        # アフィン変換の適用
        bigimg = cv2.warpAffine(bigimg, mat, (0,0),1)

    # 再び最小矩形を求めて切り出す。ただし、マージンを５つける
    _nLabels, _labelImages, data, _center = cv2.connectedComponentsWithStats(bigimg) 
    ami = np.argmax(data[1:,4])+1 # もっとも面積の大きい連結成分のラベル番号　（１のはずだが念の為）
    resultimg = bigimg[data[ami][1]-5:data[ami][1]+data[ami][3]+5,data[ami][0]-5:data[ami][0]+data[ami][2]+5]

    return resultimg

# (11) 重心から上の重心と、重心位置で断面の中心を返す関数
#   この関数ではぼかしは行わない。
def getUpperCoGandCoC(src):
    _lnum, _img, cnt, cog = cv2.connectedComponentsWithStats(src)
    ami = np.argmax(cnt[1:,4])+1 
    _c_x,c_y = np.round(cog[ami]) # 重心
    halfimg = src[:int(c_y),:].copy() # 重心位置から上を取り出す。
    _lnum, _img, cnt, cog = cv2.connectedComponentsWithStats(halfimg)
    ami =  np.argmax(cnt[1:,4])+1 
    uc_x,uc_y = np.round(cog[ami]) # 上半分の重心
    sliceindex = np.where(src[int(c_y)]!=0) # 重心断面の白画素数位置
    left = np.min(sliceindex) #  断面における最も左の白画素位置
    right = np.max(sliceindex) #  断面における最も右の白画素位置
    ccx = int((left+right)/2) #  断面中央位置
    return uc_x,uc_y,ccx,c_y


# 変数データのストアとリストア
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

