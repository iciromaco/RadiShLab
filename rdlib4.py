from glob import glob
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

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
def makethumbnail(path, imgexts=['jpg','jpge','png']):
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
    sam.save('{}THUM.PNG'.format(dirname), 'PNG')
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

# (4) 最大白領域の取り出し
def getMajorWhiteArea(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # カラーの場合はグレイ化する
    _ret,bwimg = cv2.threshold(img,128,255,cv2.THRESH_BINARY) # ２値化
    _lnum, labelimg, cnt, _cog =cv2.connectedComponentsWithStats(bwimg) # ラベリング
    areamax = np.argmax(cnt[1:,4])+1 # ０番を除く面積最大値のインデックス
    labelimg[labelimg != areamax] = 0
    labelimg[labelimg == areamax] = 255
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