import japanize_kivy
from kivy.app import App
from kivy.lang import Builder 
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty,StringProperty 
from kivy.uix.widget import Widget
from kivy.factory import Factory
from kivy.core.text import LabelBase, DEFAULT_FONT 
from kivy.utils import get_color_from_hex
from kivy.properties import BooleanProperty
# LabelBase.register(DEFAULT_FONT, "ipaexg.ttf") 
from kivy.clock import Clock
import cv2
import numpy as np
import os, sys
import threading
 
import filedialog
import rdlib4 as rd

GRC_RES ={'GRC_TEXT':['File メニューで画像を開いてください']}
DUMMYPATH = './Primrose.png'
PRIMROSE = './res/Primrose.pkl'
dummy = rd.loadPkl(PRIMROSE)

Builder.load_string('''
<MyWidget>:
    orientation: 'vertical'
    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: 1
        Spinner:
            size_hint_x: 0.25
            id: sp0
            text: 'File'
            on_text: root.do_menu()
        Label:
            size_hint_x: 0.75
            id: path0
            text: root.imgpath
    BoxLayout:
        orientation: 'vertical'
        size_hint_y: 7
        # ラベル
        Label:
            id: label1
            text_size: self.size
            text: root.res['GRC_TEXT'][0]
            halign: 'center'
            valign: 'center'
        BoxLayout:
            Button:
                id: allclear
                text: "AC"
                size_hint: (.25, 1)
            Button:
                id: eraser
                text: "Eraser"
                size_hint: (.25, 1)
            Button:
                id: framing
                text: "Framing"
                size_hint: (.5, 1)
        BoxLayout:
            Button:
                id: background0
                text: "0"
            Button:
                id: foreground1
                text: "1"
            Button:
                id: background2
                text: "2"
            Button:
                id: foreground3
                text: "3"
        BoxLayout:
            Button:
                id: rot90
                text: "Rot+90"
            Button:
                id: rot270
                text: "Rot-90"
            Button:
                id: circleS
            Button:
                id: plus
                text: "+"
                font_size: 30
            Button:
                id: minus
                text: "-"
                font_size: 40
''')
 
# opencv のカラー画像を kivy テキスチャに変換
def cv2kvtexture(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGRからRGBへ
    img2 = cv2.flip(img2,0)   # 上下反転
    height = img2.shape[0]
    width = img2.shape[1]
    texture = Texture.create(size=(width,height))
    texture.blit_buffer(img2.tostring())
    return texture

# OpenCV を用いたウィンドウ
class CV2Canvas(threading.Thread):

    def __init__(self,daemon=True):
        super(CV2Canvas,self).__init__()
        self.srcimg = rd.loadPkl(PRIMROSE)
        self.gryimg = self.makegray()
        self.outimg = self.srcimg.copy()
        self.canvas = self.makeCVcanvas(self.srcimg,self.outimg)
        cv2.namedWindow("CVCanvas",cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE)
        cv2.imshow("CVCanvas",self.canvas)
        cv2.moveWindow("CVCanvas", 100, 400)
        self.setDaemon(daemon)
        
    def makeCVcanvas(self,srcimg,outimg):
        h,w = srcimg.shape[:2]
        canvas = np.zeros((h,2*w,3),np.uint8)
        canvas[:,0:w] = srcimg
        canvas[:,w:] = outimg
        return canvas

    def loadimage(self,filepath):
        self.srcimg = cv2.imread(filepath)
        self.outimg = self.srcimg.copy()
        self.gryimg = self.makegray()
        self.canvas = self.makeCVcanvas(self.srcimg,self.outimg)       
        cv2.imshow("CVCanvas",self.canvas)
        self.needRepaint = True

    def run(self):
        global img, rorig,imgbackup,output,value,mask, rect,frame_or_mask, mouseCallBacker,filename,quitflag,framed
        framed = False
        # キーイベントループ
        while(1):
            if k == ord('0'): # 背景領域の指定
            # print(" 背景領域を指定 \n")
                value = DRAW_BG
            elif k == ord('1'): # 対象の指定
                # print(" 切り出し対象領域を指定 \n")
                value = DRAW_FG
            elif k == ord('2'): # 背景かも知れない領域の指定
                # print(" 背景かも知れない領域の指定 \n")
                value = DRAW_PR_BG
            elif k == ord('3'): # 前景かもしれない領域の指定
                # print(" 前景かもしれない領域の指定 \n")
                value = DRAW_PR_FG
        
            elif k == ord('+'):
                mouseCallBacker.thicknessUp()

            elif k == ord('-'):
                mouseCallBacker.thicknessDown()
                
            elif k == ord('9'): # 90度回転
                    if not mouseCallBacker.framed:
                        print(" 回転します\n")
                                        
                        rorig = rorig.transpose(1,0,2)[::-1,:,:]
                        img = rorig.copy() 
                        img, imgbackup,halfimg,output,halfoutput,mask= prepareimg(img,size = IMAGESIZE)                    
                        # rect= (rect[1],rect[0],rect[3],rect[2]) 
                        makeupAndShowImage() 
                    else:
                        print(" フレーム確定後は回転できません。リセットしてください。\n")            
                    
            
            elif k == ord('s'): # 画像の保存
                bar = np.zeros((img.shape[0],5,3),np.uint8)
                res = np.hstack(( imgbackup,bar,img,bar,output))
                
                print("抽出結果を保存するパスを選んで下さい（拡張子は不要）")
                savepath = saveFilePath(filename)
                
                _ret,bw = cv2.threshold(cv2.cvtColor(output,cv2.COLOR_BGR2GRAY),1,255,cv2.THRESH_BINARY)

                savedir, ext = os.path.splitext(savepath)
                resultimg, (x0,y0,w0,h0),m = cutmargin(bw,margin=5)
                cv2.imwrite(savepath,resultimg)
                timg = np.zeros((h0+2*m,w0+2*m,3),np.uint8)
                timg[m:m+h0,m:m+w0]=img[y0:y0+h0,x0:x0+w0]
                cv2.imwrite(savedir+"Color.png",timg)
                print(savepath,"に保存しました。")
                
                cv2.imwrite('grabcut_output.png',res)
                print("抽出結果は保存先:{}に、\n, それとは別に合成画像を grabcut_output.png に結果を保存しました.\n".format(savepath+".png"))
                quitflag = False
                break
                
            elif k == ord('r'): # reset everything
                print("リセット \n")
                mouseCallBacker.init()
                rorig = orig.copy()
                img, imgbackup,halfimg,output,halfoutput,mask,ratio = prepareimg(orig,size = IMAGESIZE)
                makeupAndShowImage()
                
            elif k == 13 : #  Enter キー  セグメンテーションの実行
                print("セグメンテーションの実行中。新しいメッセージが表示されるまでお待ち下さい。 \n")
                if (frame_or_mask == 0):         # grabcut with rect
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgdmodel = np.zeros((1,65),np.float64)
                    mask = mask.copy()
                    cv2.grabCut( imgbackup,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                    frame_or_mask = 1
                elif frame_or_mask == 1:         # grabcut with mask
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgdmodel = np.zeros((1,65),np.float64)
                    cv2.grabCut( imgbackup,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
                print(" 抽出がうまくいっていない場合は、手動でタッチアップしてから再度 Enter  を押して下さい。\n ０、２　背景領域の指定、１，３ 抽出対象領域の指定 \n")
                framed = True
                
            mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
            output = cv2.bitwise_and( imgbackup, imgbackup,mask=mask2)
            if gusecolor == False:
                _ret,bw = cv2.threshold(cv2.cvtColor(output,cv2.COLOR_BGR2GRAY),1,255,cv2.THRESH_BINARY)
                output =  cv2.cvtColor(bw,cv2.COLOR_GRAY2BGR)
            makeupAndShowImage()


        # ３または４チャネル画像をdsグレイ化
    def makegray(self):
            img = self.srcimg
            if len(img.shape) == 3 :
                if img.shape[2] == 4: # Alpha チャネル付き
                    gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
                else:
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            return gray

# インタフェースパレット
from kivy.graphics.texture import Texture
class MyWidget(BoxLayout):
    mode = 'None'
    res = GRC_RES
    imgpath = DUMMYPATH
    quitflag = False
    def __init__(self,app,**kwargs):
        super(MyWidget,self).__init__(**kwargs)
        self.app = app
        self.cv2canvas = CV2Canvas(daemon=True)
        self.cv2canvas.start()

    # メニュー処理
    def do_menu(self):
        if self.ids['sp0'].text == 'File':
            return
        else:
            self.mode = self.ids['sp0'].text
            self.ids['sp0'].text = 'File'
        if self.mode == 'Open':
            self.show_load()
        if self.mode == 'Save':
            self.show_save()
        if self.mode == 'Quit':
            sys.exit()

    def dismiss_popup(self):
        self._popup.dismiss()
        Window.size = self.keepsize
 
    def show_load(self):
        self.keepsize = Window.size
        Window.size = (600,600)
        content = Factory.LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9,0.9))
        self._popup.open()
 
    def load(self, filepath):
        self.ids['path0'].text = filepath
        self.cv2canvas.loadimage(filepath)
        self.dismiss_popup()
 
    def show_save(self):
        self.keepsize = Window.size
        Window.size = (600,600)
        content = Factory.SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
 
    def save(self, path, filename):
        path = os.path.join(path, filename)
        path1 = os.path.splitext(path)
        if not path1[1].lower() in ['.png','.jpg']:
            path = path1[0]+'.png'
        self.ids['path0'].text = path
        cv2.imwrite(path,self.outimg)
        self.dismiss_popup()

PENSIZE = 5
BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
MAGENTA = [255,0,255]    # sure BG
BLACK = [0,0,0]
WHITE = [255,255,255]   # sure FG
MINRECTSIZE = 400 # 領域指定とそうでない操作の切り分けのための矩形面積の下限

MAXIMAGESIZE = 1024 # 強制的に画像サイズをの数字以下に縮小する。
WINDOWSSIZE = MAXIMAGESIZE//2 # 表示ウィンドウサイズ
NEEDSIZE =256 # 対象に要求するサイズ。矩形がこれ以下であればこの値以上になるように解像度を上げて GrabCut する

DRAW_BG = {'color' : MAGENTA, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

def mkpensizeSample(size=5):
    pensizeCanvas = np.zeros((32,32),np.uint8)
    cv2.circle(pensizeCanvas,(16,16),size,255,-1)
    imagetexture = cv2kvtexture(pensizeCanvas)
    return imagetexture

# 画像が大きすぎる場合、IMAGESIZE以下になるように縮小する
def prepareimg(img, limitsize = MAXIMAGESIZE):
        img, halfimg = resize(img, limitsize=size)   # 縦横いずれもlimitsize以下になるよう縮小
        imgbackup = img.copy()                        # a copy of resized image
        mask = np.zeros(img.shape[:2],np.uint8)       # for mask initialized to PR_BG
        output = np.zeros(img.shape,np.uint8)          # for output image to be shown
        halfoutput = np.zeros(halfimg.shape,np.uint8)  # for halfsize output image 
        return img, imgbackup,halfimg,output,halfoutput,mask

# 画像が大きすぎる場合に、制限サイズ内に収まるよう縮小するとともに表示用ハーフサイズ画像を生成
def resize(img, limitsize=MAXIMAGESIZE):
    height,width = img.shape[:2]
    maxsize = max(height,width)
    if maxsize > limitsize:
        heightN = limitsize*height//maxsize
        width = limitsize*width//maxsize
    # 切り出し対象が枠いっぱいだった時のために少しマージンをつける。
    output = np.zeros((height+80, width+80,3),np.uint8) 
    output[40:40+height,40:40+width]=cv2.resize(img,(width,height))
    h2 = output.shape[0]//2
    w2 = output.shape[1]//2
    halfimg=cv2.resize(output,(w2,h2))
    return output,halfimg

# カット
def cutImage(srcimg, usecolor=False):
    global img, imgbackup,orig,rorig,output,value,mask,shrinkN,filename, mouseCallBacker,gusecolor
    
    gusecolor = usecolor
    
    # print( __doc__ )
    # print("画像ファイルを選んで下さい")
    
    # readpath = readFilePath()
    # readdir,filename = os.path.split(readpath)
    
    orig = srcimg.copy() # オリジナルを複製
    rorig = srcimg.copy() # 回転用のオリジナルの複製
    img = orig.copy() # 操作用画像
    # 操作用画像
    img, imgbackup,halfimg,output,halfoutput,mask= prepareimg(img,limitsize = MAXIMAGESIZE)
   
    # input and output windows
    # cv2.namedWindow('output')
    # cv2.namedWindow('input')
    # cv2.imshow('output',halfoutput)
    # cv2.imshow('input', halfimg)

    mouseCallBacker = myMouse('input')
 
    # cv2.moveWindow('input',0,90)
    # cv2.moveWindow('output',halfimg.shape[1],90)
    
    # print(" マウスの左ドラッグで抽出対象を囲って下さい \n")

    do_keyEventLoop()
    
    # cv2.destroyAllWindows()
    # cv2.waitKey(1) # jupyter notebook の場合、destroyWindowの後に waitKey しないと終了できなくなる

# マウス操作
class myMouse:
    def __init__(self, windowname):
        self.init()
        cv2.setMouseCallback(windowname, self.callBack, None)
        
    def init(self):
        global value,mask,framing,drawing,rect,frame_or_mask
        # setting up flags
        rect = (0,0,1,1)
        drawing = False         # 描画モードオン
        framing = False           # 選択枠設定中
        self.framed = False       # 枠設定は完了している
        frame_or_mask = 100      # flag for selecting rect or mask mode
        value = DRAW_BG         # drawing initialized to FG
        self.thickness = PENSIZE           # ブラシサイズ
    
    def thicknessUp(self):
        self.thickness +=1
        
    def thicknessDown(self):
        if self.thickness > 0:
            self.thickness -=1
        
    def callBack(self, event, x, y, flags, param=None) :
        global img, imgbackup,halfimg,mask,output,value,mask,framing,framed,drawing,rect,frame_or_mask
        # フレーム設定フェーズの処理
        if self.framed == False:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.lx,self.ly = 2*x,2*y
                framing = True # 矩形描画モードオン
            elif event == cv2.EVENT_MOUSEMOVE:  
                if framing == True :                      
                    img =  imgbackup.copy()
                    cv2.rectangle(img,(self.lx,self.ly),(2*x,2*y),BLUE,4)
                    rect = (min(self.lx,2*x),min(self.ly,2*y),abs(self.lx-2*x),abs(self.ly-2*y))
            elif event == cv2.EVENT_LBUTTONUP:
                framing = False
                tmps = abs(self.lx-2*x)*abs(self.ly-2*y)  # 指定矩形の面積
                if tmps > MINRECTSIZE:
                    self.framed = True
                    cv2.rectangle(img,(self.lx,self.ly),(2*x,2*y),BLUE,2)
                    rect0 = (min(self.lx,2*x),min(self.ly,2*y),abs(self.lx-2*x),abs(self.ly-2*y))
                    img, imgbackup,halfimg,output,halfoutput,mask = prepareimg(rorig,size = IMAGESIZE)
                    cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),BLUE,2)
                    makeupAndShowImage()
                    frame_or_mask = 0
                    print( "Enterキーを押せば抽出を始めます。終わるまでしばらくお待ち下さい \n")
        else: # 枠指定がすでに済んでいる場合
            imgroi = img[rect[1]-20:rect[1]+rect[3]+20,rect[0]-20:rect[0]+rect[2]+20]
            maskroi = mask[rect[1]-20:rect[1]+rect[3]+20,rect[0]-20:rect[0]+rect[2]+20]
            if drawing == True:
                if event == cv2.EVENT_MOUSEMOVE:
                    cv2.circle(imgroi,(x,y),self.thickness,value['color'],-1)cv2
                    cv2.circle(maskroi,(x,y),self.thickness,value['val'],-1)
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
            else:
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
        
        makeupAndShowImage()

# アプリケーションメイン 
class MyApp(App):
    def build(self):
        Window.size = (300,200)
        Window.Pos = (0,0)
        mywidget = MyWidget(self)
        mywidget.ids['sp0'].values = ('Open','Save','Quit')
        mywidget.ids['circleS']
        self.title = 'GrabCut'
        return mywidget

MyApp().run()


