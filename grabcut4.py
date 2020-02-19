import japanize_kivy

from kivy.app import App
from kivy.lang import Builder 
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty,StringProperty, NumericProperty
from kivy.uix.widget import Widget
from kivy.factory import Factory
from kivy.core.text import LabelBase, DEFAULT_FONT 
from kivy.utils import get_color_from_hex
from kivy.properties import BooleanProperty
from kivy.graphics.texture import Texture
from kivy.clock import Clock

import cv2
import numpy as np
import os, sys
import threading
 
import filedialog
import rdlib4 as rd

from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle, Point, GraphicException
from math import sqrt
import time

# 右クリックで表示される赤丸を禁止
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,disable_multitouch') 

GRC_RES ={'GRC_TEXT':['File メニューで画像を開いてください']}
DUMMYPATH = './Primrose.png'
PRIMROSE = './res/Primrose.pkl'
IWINSIZE = (854,704)
BUTTONH = 32
DUMMYIMG = rd.loadPkl(PRIMROSE)
picdic = rd.loadPkl('./res/picdic.pkl')

Builder.load_string('''
#:set BH 32
#:set WINW 854
#:set WINH 704
#:set IMGH 640
#:set IMGW 427
<MyWidget>:
    size_hint: None,None
    size: WINW, WINH
    CanvasFloatLayout:
        id: rdcanvas
        size_hint: None,None
        size: self.parent.size
        BoxLayout:
            orientation: 'horizontal'
            size_hint: None,None
            size: WINW, IMGH
            pos: 0, BH
            canvas:
                Color:
                    rgb: 0, 0, 0
                Rectangle:
                    size: self.size
            Image:
                id: srcimg
                size_hint: None,None
                size: IMGW,IMGH
                texture: root.srctexture
                allow_stretch:True
            Image:
                id: outimg
                size_hint: None,None
                size: IMGW,IMGH
                texture: root.outtexture
    FloatLayout:
        size_hint: None,None
        size: WINW,BH
        pos: 0,WINH-BH        
        BoxLayout:
            id: topmenu
            orientation: 'horizontal'
            #pos: 0,WINH-BH
            Spinner:
                size_hint_x: 0.2
                id: sp0
                text: 'File'
                on_text: root.do_menu()
            Label:
                size_hint_x: 1
                id: message
                text: root.res['GRC_TEXT'][0]
                halign: 'center'
                valign: 'center'                
    FloatLayout
        size_hint: None,None
        size: WINW,BH
        pos: 0,0
        BoxLayout:
            id: buttonmenu
            orientation: 'horizontal'
            TextInput:
                id: path0
                text: root.imgpath
                font_size: 12
                size_hint_x: 0.8
            BoxLayout:
                size_hint_x: 1.0
                orientation: 'horizontal'
                Button:
                    id: allclear
                    text: "AC"
                Button:
                    id: eraser
                    text: "ER"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['eraser']
                Button:
                    id: framing
                    text: "FR"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['frame']
                Button:
                    id: background0
                    text: "0"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['zero']
                Button:
                    id: foreground1
                    text: "1"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['one']
                Button:
                    id: background2
                    text: "2"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['two']
                Button:
                    id: foreground3
                    text: "3"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['three']
                Button:
                    id: rot90
                    text: "R+90"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['rot90']
                Button:
                    id: rot270
                    text: "R-90"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['rot270']
                Button:
                    id: plus
                    text: "+"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['plus']
                Button:
                    id: minus
                    text: "+"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['minus']
                Button:
                    id: cut
                    text: "-"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['cut']
''')


def calculate_points(x1, y1, x2, y2, steps=5):
    dx = x2 - x1
    dy = y2 - y1
    dist = sqrt(dx * dx + dy * dy)
    if dist < steps:
        return
    o = []
    m = dist / steps
    for i in range(1, int(m)):
        mi = i / m
        lastx = x1 + dx * mi
        lasty = y1 + dy * mi
        o.extend([lastx, lasty])
    return o


class CanvasFloatLayout(FloatLayout):
    
    def on_touch_down(self, touch):
        if Widget.on_touch_down(self, touch):
            return

        print(self.collide_widget)
        print("down",touch.pos,self.to_window(touch.pos[0],touch.pos[1]))
        win = self.get_parent_window()
        ud = touch.ud
        ud['group'] = g = str(touch.uid)
        pointsize = 5
        ud['color'] = 1

        with self.canvas:
            Color(ud['color'], 1, 1, mode='hsv', group=g)
            ud['lines'] = [
                Rectangle(pos=(touch.x, 0), size=(1, win.height), group=g), # クロスカーソル 縦
                Rectangle(pos=(0, touch.y), size=(win.width, 1), group=g), # クロスカーソル　横
                Point(points=(touch.x, touch.y), source='res/picdicpics/particle.png',
                                       pointsize=pointsize, group=g)
                ]

        ud['label'] = Label(size_hint=(None, None))
        self.update_touch_label(ud['label'], touch)
        self.add_widget(ud['label'])
        touch.grab(self)
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return
        ud = touch.ud
        ud['lines'][0].pos = touch.x, 0
        ud['lines'][1].pos = 0, touch.y

        index = -1

        while True:
            try:
                points = ud['lines'][index].points
                oldx, oldy = points[-2], points[-1]
                break
            except:
                index -= 1

        points = calculate_points(oldx, oldy, touch.x, touch.y)

        if points:
            try:
                lp = ud['lines'][-1].add_point
                for idx in range(0, len(points), 2):
                    lp(points[idx], points[idx + 1])
            except GraphicException:
                pass

        ud['label'].pos = touch.pos

        t = int(time.time())
        if t not in ud:
            ud[t] = 1
        else:
            ud[t] += 1
        self.update_touch_label(ud['label'], touch)

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return    
        touch.ungrab(self)
        ud = touch.ud
        self.canvas.remove_group(ud['group'])
        self.remove_widget(ud['label'])
    
    def update_touch_label(self, label, touch):
        label.text = 'ID: %s\nPos: (%d, %d)\nClass: %s' % (
            touch.id, touch.x, touch.y, touch.__class__.__name__)
        label.texture_update()
        label.pos = touch.pos
        label.size = label.texture_size[0] + 20, label.texture_size[1] + 20
    
# インタフェースパレット

# opencv のカラー画像を kivy テキスチャに変換
def cv2kvtexture(img):
    if len(img.shape) == 2:
        img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    else:
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGRからRGBへ
    img2 = cv2.flip(img2,0)   # 上下反転
    height = img2.shape[0]
    width = img2.shape[1]
    texture = Texture.create(size=(width,height))
    texture.blit_buffer(img2.tostring())
    return texture

class MyWidget(BoxLayout):
    windowsize = DUMMYIMG.shape[1]*2, DUMMYIMG.shape[0]+2*BUTTONH
    srctexture = cv2kvtexture(DUMMYIMG)
    outtexture = srctexture
    res = GRC_RES
    imgpath = DUMMYPATH
    m_size = BUTTONH
    pictexture = {key:cv2kvtexture(picdic[key]) for key in picdic}
    def __init__(self,**kwargs):
        super(MyWidget,self).__init__(**kwargs)
        self.setsrcimg(DUMMYIMG)
        self.rdcanvas = CanvasFloatLayout()
        self.ids['rdcanvas'] = self.rdcanvas
        Clock.schedule_interval(self.update, 1)

    def update(self, dt):
        print(self.srcimg.shape,self.windowsize)
        Window.size = self.windowsize

    def setsrcimg(self,srcimg):
        self.srcimg = srcimg
        self.srctexture = cv2kvtexture(srcimg)
        self.ids['srcimg'].texture = self.srctexture
        self.outtexture = self.afterload()
        self.windowsize = self.srcimg.shape[1]*2, self.srcimg.shape[0]+2*BUTTONH
        Window.size = self.windowsize
        self.ids['rdcanvas'].size = self.srcimg.shape[1]*2,self.srcimg.shape[0]
        self.ids['srcimg'].size = self.srcimg.shape[1],self.srcimg.shape[0]
        self.ids['outimg'].size = self.srcimg.shape[1],self.srcimg.shape[0]
        self.ids['topmenu'].size = self.windowsize[0],BUTTONH
        self.ids['topmenu'].pos = 0,self.srcimg.shape[0]+BUTTONH
        self.ids['buttonmenu'].size = self.windowsize[0],BUTTONH

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
        srcimg = cv2.imread(filepath)
        self.setsrcimg(srcimg)
        # self.ids['srcimg'].texture = cv2kvtexture(self.srcimg)
        # self.afterload()
        self.dismiss_popup()

    def afterload(self):
        self.outimg = self.srcimg.copy()
        self.gryimg = self.makegray()
        self.ids['outimg'].texture = cv2kvtexture(self.gryimg)
        # self.cv2canvas.loadimage(filepath)

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

    def makegray(self):
        img = self.srcimg
        if len(img.shape) == 3 :
            if img.shape[2] == 4: # Alpha チャネル付き
                gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return gray

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
                    cv2.circle(imgroi,(x,y),self.thickness,value['color'],-1)
                    cv2.circle(maskroi,(x,y),self.thickness,value['val'],-1)
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
            else:
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
        
        makeupAndShowImage()

'''
# アプリケーションメイン 
class MyApp(App):
    def build(self):
        mywidget = MyWidget(self)
        mywidget.ids['sp0'].values = ('Open','Save','Quit')
        self.title = 'GrabCut'
        return mywidget

if __name__ == '__main__':
    MyApp().run()
'''
class MyApp(App):
    title = 'Touchtracer'

    def build(self):
        mywidget = MyWidget()
        mywidget.ids['sp0'].values = ('Open','Save','Quit')
        self.title = 'GrabCut'
        return mywidget

    def on_pause(self):
        return True

if __name__ == '__main__':
   MyApp().run()