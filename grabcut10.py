import cv2
import numpy as np
import os, sys
 
import japanize_kivy  #  pip install japanize_kivy

from kivy.app import App
from kivy.lang import Builder 
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.factory import Factory
from kivy.graphics import Color, Rectangle, Point, Ellipse,GraphicException
from kivy.uix.label import Label
from kivy.graphics.vertex_instructions import Line


'''
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty,StringProperty, NumericProperty, BooleanProperty
from kivy.core.text import LabelBase, DEFAULT_FONT 
from kivy.utils import get_color_from_hex
from kivy.properties import 
'''

import filedialog
import rdlib4 as rd

# 右クリックで表示される赤丸を禁止
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,disable_multitouch') 

GRC_RES ={
'OpenImage':'File メニューで画像を開いてください',
'TopLeft':'対象を枠で囲み指定します。左上の点を指定してください',
'BottomRight':'対象を枠で囲み指定します。右下の点を指定してください',
'Confirm':'選択できたらCutボタンを押してください',
'OnCutting':'カット中です。しばらくお待ちください',
'Finished':'満足できるまで何度かCutをクリック or 1234でヒント情報を追加してCut',
'Marking0':'Mark sure BG 確実に背景となる領域をマーク',
'Marking1':'Mark sure FG 確実に対象である領域をマーク',
'Marking2':'Mark probably BG 背景画素の多い領域をマーク',
'Marking3':'Mark probably FG 前景がその多い領域ををマーク'
}

DUMMYPATH = './Primrose.png'
PRIMROSE = './res/Primrose.pkl'
BUTTONH = 32
DUMMYIMG = rd.loadPkl(PRIMROSE)
picdic = rd.loadPkl('./res/picdic.pkl')

PENSIZE = 5
BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
MAGENTA = [255,0,255]    # sure BG
BLACK = [0,0,0]
WHITE = [255,255,255]   # sure FG
from kivy.utils import get_color_from_hex
C4 = [
    get_color_from_hex('#FF00FF'), # MAGENDA for BG
    get_color_from_hex('#FFFFFF'), # WHITE for FG
    get_color_from_hex('#FF0000'), # RED for FG
    get_color_from_hex('#00FF00')  # GREEN for BG 
]
COLORS = []
for i in range(4):
    cc = C4[i]
    item = 'Color({},{},{},mode="rgb",group="{}")'.format(cc[0],cc[1],cc[2],str(i))
    COLORS.append(item)

MINRECTSIZE = 400 # 領域指定とそうでない操作の切り分けのための矩形面積の下限

MAXIMAGESIZE = 1024 # 強制的に画像サイズをの数字以下に縮小する。
WINDOWSSIZE = MAXIMAGESIZE//2 # 表示ウィンドウサイズ
NEEDSIZE =256 # 対象に要求するサイズ。矩形がこれ以下であればこの値以上になるように解像度を上げて GrabCut する

DRAW_BG = {'color' : MAGENTA, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_BG = {'color' : RED, 'val' : 2}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_COLORS = [DRAW_BG,DRAW_FG,DRAW_PR_BG,DRAW_PR_FG]

IF_H = 32 #  ボタンやメニューの高さ

Builder.load_string('''
#:set BH 32
#:set dummypath './Primrose.png'
<MyWidget>:
    FloatLayout:
        BoxLayout:
            size_hint: None,None
            size: root.width,BH
            pos_hint: {"x":0, "top":1}
            BoxLayout:
                id: topmenu
                orientation: 'horizontal'
                pos: 0, root.top
                Spinner:
                    size_hint_x: 0.2
                    id: sp0
                    text: 'File'
                    on_text: root.do_menu()
                Label:
                    size_hint_x: 1
                    id: message
                    halign: 'center'
                    valign: 'center'
        BoxLayout:
            id: rdcanvas
            orientation: 'horizontal'
            size_hint: None,None
            size: root.width,root.height-2*BH
            pos: 0,BH
            Image:
                id: srcimg
                size_hint: None,None
                size: self.parent.width/2,self.parent.height
            Image:
                id: outimg
                size_hint: None,None
                size: self.parent.width/2,self.parent.height
        BoxLayout:
            id: buttonmenu
            size_hint: None,None
            size: root.width,BH
            pos: 0,0
            orientation: 'horizontal'
            TextInput:
                id: path0
                text: dummypath
                font_size: 12
                size_hint_x: 0.8
            BoxLayout:
                size_hint_x: 1.0
                orientation: 'horizontal'
                Button:
                    id: allclear
                    text: "AC"
                    on_press: root.resetAll()
                ToggleButton:
                    id: eraser
                    text: "ER"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['eraser']
                ToggleButton:
                    id: framing
                    text: "FR"
                    group: "mode"
                    state : "down"
                    on_press: root.startFraming()
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['frame']
                ToggleButton:
                    id: mark0 # Sure Background
                    text: "0"
                    group: "mode"
                    on_press: root.on_markup(0)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['zero']
                ToggleButton:
                    id: mark1 # Sure Foreground
                    text: "1"
                    group: "mode"
                    on_press: root.on_markup(1)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['one']
                ToggleButton:
                    id: mark2 # Probably Background
                    text: "2"
                    group: "mode"
                    on_press: root.on_markup(2)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['two']
                ToggleButton:
                    id: mark3 # Probably Foreground
                    text: "3"
                    group: "mode"
                    on_press: root.on_markup(3)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['three']
                Button:
                    id: rot90
                    text: "R+"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['rot90']
                Button:
                    id: rot270
                    text: "R-"
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['rot270']
                Button:
                    id: plus
                    text: "+"
                    on_press: root.thicknessUpDown(1)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['plus']
                BoxLayout:
                    size_hint: None,None
                    size: BH,BH
                    Image:
                        id: dotsize
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: 
                Button:
                    id: minus
                    text: "-"
                    on_press: root.thicknessUpDown(-1)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['minus']
                ToggleButton:
                    id: rdreform # reform flag
                    text: "RF"
                    group: "rdreform"
                    state: "down"
                Button:
                    id: cut
                    text: "-"
                    on_press: root.grabcut()
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['cut']

''')

# 2点を結ぶ線分を、間隔 step で分割した、両端以外の点列[x1,y1,x2,y2,...]を返す 
def calculate_points(x1, y1, x2, y2, steps=5):
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx * dx + dy * dy)
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

# opencv のカラー画像を kivy テキスチャに変換
def cv2kvtexture(img,force3 = True):
    if len(img.shape) == 2:
        img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3 or force3:
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGRからRGBへ
    else : # with alpha
        img2 = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) # BGRAからRGBAへ
    img2 = cv2.flip(img2,0)   # 上下反転
    height = img2.shape[0]
    width = img2.shape[1]
    texture = Texture.create(size=(width,height))
    colorfmt = 'rgb' if img2.shape[2]==3 else 'rgba'
    texture.blit_buffer(img2.tostring(), colorfmt=colorfmt)
    return texture

# インタフェースパレット
class MyWidget(BoxLayout):
    windowsize = DUMMYIMG.shape[1]*2, DUMMYIMG.shape[0]+2*BUTTONH # 初期ウィンドウサイズ
    pictexture = {key:cv2kvtexture(picdic[key]) for key in picdic}
    touchud = [] # touch.ud の記憶場所

    def __init__(self,**kwargs):
        super(MyWidget,self).__init__(**kwargs)
        self.fState = 0 # 枠指定の状態 0:初期、1:1点指定済み、2:指定完了
        self.setsrcimg(DUMMYIMG)
        self.rect = [0,0,1,1] # 切り出し枠
        self.fp1 = [0,0] # 切り出し枠枠の1点目の座標
        self.pointsize = 5 # ペンサイズ
        self.pensizeimage()
        self.mask = None # grabcut 用のmask
        self.ids['message'].text = GRC_RES['OpenImage']
        Clock.schedule_interval(self.update, 0.1) # ウィンドウサイズの監視固定化

    # 入力画像をセット
    def setsrcimg(self,srcimg):
        self.srcimg = srcimg
        h,w = srcimg.shape[:2]
        self.srctexture = cv2kvtexture(srcimg)
        self.ids['srcimg'].texture = self.srctexture
        self.resetAll()

    # 画像読み込み後の初期化
    def resetAll(self):
        srcimg = self.srcimg
        for i in range(4):
            self.canvas.remove_group(str(i))  # 枠線および描画線消去
        self.workmg = srcimg.copy() # 作業用＝左ペインに表示
        self.restoreimg = srcimg.copy() # １手戻る用
        gryimg = cv2.cvtColor(self.srcimg,cv2.COLOR_BGR2GRAY) # グレイ画像作成
        self.silhouette = mask = rd.getMajorWhiteArea(gryimg,binary=True) # シルエット画像作成
        img4 = cv2.cvtColor(self.srcimg,cv2.COLOR_BGR2BGRA) # アルファチャネル追加
        img4[:,:,3] = (127*(mask//255)) +128 # 黒領域の透明化
        self.outimg = rd.draw2(mask,img4,thickness=1,color=(0,0,200,255)) # 輪郭描画
        self.ids['outimg'].texture = cv2kvtexture(self.outimg,force3=False) # 仮結果画像
        self.ids['message'].text =  GRC_RES['TopLeft'] # メッセージ表示

        self.fState = 0 # 枠指定の状態初期化
        self.frame_or_mask = 0 # 0 -> mask は初期状態 1 -> セット済み
        self.ids['framing'].state = "down"

        self.canvas.remove_group('0')
        self.canvas.remove_group('1')

    def pensizeimage(self):
        pensize = self.pointsize
        pimg = np.zeros((32,32,4),np.uint8)
        cv2.circle(pimg,(16,16),pensize,(255,255,255,255),-1)
        ptxt = cv2kvtexture(pimg,force3=False)
        self.ids['dotsize'].texture = ptxt

    # ウィンドウサイズを固定化
    def update(self, dt):
        h,w = self.srcimg.shape[:2]
        Window.size = (2*w,h+2*IF_H)

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

    # ファイルウィンドウをポップした状態からの復帰
    def dismiss_popup(self):
        self._popup.dismiss()
        Window.size = self.keepsize
 
    # ファイルの選択と読み込み
    def show_load(self):
        def load(filepath):
            self.ids['path0'].text = filepath
            srcimg = rd.imread(filepath)
            self.setsrcimg(srcimg)
            self.dismiss_popup()

        self.keepsize = Window.size
        Window.size = (600,600)
        content = Factory.LoadDialog(load=load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9,0.9))
        self._popup.open()

    def show_save(self):
        def save(path, filename):
            path = os.path.join(path, filename)
            path1 = os.path.splitext(path)
            if not path1[1].lower() in ['.png','.jpg']:
                path = path1[0]+'.png'
            self.ids['path0'].text = path
            rd.imwrite(path,self.silhouette)

            self.dismiss_popup()

        self.keepsize = Window.size
        Window.size = (600,600)
        content = Factory.SaveDialog(save=save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    # マウスカーソルが入力画像内にあるかどうかのテスト
    def isInCanvas(self,touch):
        x = touch.x
        y = touch.y
        h,w = self.srcimg.shape[:2]
        return y >= BUTTONH and y < h + BUTTONH # and x < w

    # 枠付けの開始
    def startFraming(self):
        if self.fState == 2:
            self.ids['framing'].state = "normal"

    # 枠付け中であるかどうかの判定
    def nowFraming(self):
        return self.fState < 2 and self.ids['framing'].state == "down" 

    # マーキング
    def on_markup(self,ret):
        if self.fState < 2:
            self.ids['mark%d' % ret].state = "normal"
            self.ids['framing'].state = "down"
        else:
            self.ids['message'].text = GRC_RES['Marking%d' % (ret)]

    # マーキング中であるかどうかの判定
    def nowMarking(self):
        ret = -1
        for n in range(4):
            if self.ids['mark%d' % n].state == 'down':
                ret = n
        return ret

    # ヒント情報の描画
    def drawPoint(self,points,colorvalue):
        self.restoreimg = self.srcimg.copy()
        for idx in range(0,len(points),2):
            x = int(points[idx])
            y = self.srcimg.shape[0]+BUTTONH-int(points[idx+1])
            # cv2.circle(self.workimg,(x,y),self.pointsize,colorvalue['color'],-1)
            cv2.circle(self.mask,(x,y),self.pointsize,colorvalue['val'],-1)

    # ペンサイズの増減
    def thicknessUpDown(self, diff):
        pointsize = self.pointsize + diff
        if pointsize > 0 and pointsize < 31: 
            self.pointsize = pointsize
        self.pensizeimage()

    # マウスイベントの処理
    def on_touch_down(self, touch):
        if Widget.on_touch_down(self, touch) or not self.isInCanvas(touch): 
            # ボタンなどのウィジット上でタッチした場合は処理をスルー
            return
        
        h,w = self.srcimg.shape[:2]
        ud = touch.ud
        x = touch.x if touch.x < w else touch.x - w
        
        if self.nowFraming(): # 枠設定中
            g = str(self.fState) 
            with self.canvas:
                # Color(ud['color'], 1, 1, mode='hsv', group=g)
                Color(0, 1, 0, mode='rgba',group=g)
                ud['cross'] = [
                    Rectangle(pos=(x, BUTTONH), size=(1, h), group=g), # クロスカーソル 縦
                    Rectangle(pos=(0, touch.y), size=(2*w, 1), group=g), # クロスカーソル　横
                    Rectangle(pos=(x+w, BUTTONH), size=(1, h), group=g)]
            ud['label'] = Label(size_hint=(None, None))
            self.add_widget(ud['label'])
            self.update_touch_label(ud['label'], touch)
        else:
            mark = self.nowMarking()
            if mark < 0: # not on marking
                return

            ud['group'] = g = str(mark) 
            ps = self.pointsize 

            with self.canvas:
                exec(COLORS[mark])
                # ud['drawings'] = Point(points=(touch.x, touch.y), source='res/picdicpics/particle.png',
                ud['drawings'] = Point(points=(x, touch.y), source='res/picdicpics/pennib.png',
                                      pointsize=ps, group=g) # # 
                self.drawPoint(ud['drawings'].points,colorvalue=DRAW_COLORS[mark])

        touch.grab(self) # ドラッグの追跡を指定            
        return True

    def on_touch_move(self,touch):
        # 入力画像内でドラッグが開始されて、継続して画像内でドラッグが続いているかどうかのチェック
        if (touch.grab_current is not self) or (not self.isInCanvas(touch)):
            return

        h,w = self.srcimg.shape[:2]
        ud = touch.ud
        x = touch.x if touch.x < w else touch.x - w

        if self.nowFraming(): # 枠設定中
            
            ud['cross'][0].pos = x, BUTTONH
            ud['cross'][1].pos = 0, touch.y
            ud['cross'][2].pos = x + w, BUTTONH                
            self.update_touch_label(ud['label'], touch)
        else:
            mark = self.nowMarking()
            if mark < 0:
                return

            ud['group'] = g = str(mark) 
            ps = self.pointsize

            while True:
                try:
                    pts = ud['drawings'].points
                    oldx, oldy = pts[-2], pts[-1]
                    break
                except:
                    index -= 1

            # カーソル移動が早すぎて間が飛んだ時のための補間処理
            points = calculate_points(oldx, oldy, x, touch.y, steps=ps)
            if points:
                try:
                    lp = ud['drawings'].add_point # add_point関数 を lp と alias している
                    for idx in range(0, len(points), 2):
                        lp(points[idx], points[idx + 1])
                except GraphicException:
                    pass

                self.drawPoint(ud['drawings'].points,colorvalue=DRAW_COLORS[mark])
            
    def on_touch_up(self, touch):
        if touch.grab_current is not self or (not self.isInCanvas(touch)):
            return    

        h,w = self.srcimg.shape[:2]
        ud = touch.ud

        if self.nowFraming(): # 枠設定中
            if self.fState == 0: # １点目未設定
                self.fp1[0] = ud['cross'][0].pos[0]
                self.fp1[1] = (h+BUTTONH)-ud['cross'][1].pos[1]
                self.ids['message'].text =  GRC_RES['BottomRight']
                self.fState = 1 # 1点目確定
            elif self.fState == 1:
                p2x = ud['cross'][0].pos[0]
                p2y = (h+BUTTONH)-ud['cross'][1].pos[1]
                self.rect = (min(self.fp1[0],p2x),min(self.fp1[1],p2y),abs(self.fp1[0]-p2x),abs(self.fp1[1]-p2y))
                self.ids['message'].text =  GRC_RES['Confirm']
                self.ids['framing'].state = "normal"
                self.fState = 2
            self.remove_widget(ud['label'])    
        else:
            touch.ungrab(self)
        
        testimg = self.export_to_png("__tmp.png")
        cv2.imwrite("__cvimg.png",self.srcimg)

    # 座標表示
    def update_touch_label(self, label, touch):
        h,w = self.srcimg.shape[:2]
        y = (h + BUTTONH) - touch.y
        label.text = '(%d, %d)' % (touch.x, y)
        label.texture_update()
        label.pos = touch.pos
        label.size = label.texture_size[0] + 20, label.texture_size[1] + 20

    # セグメンテーション
    def grabcut(self):
        if self.fState < 2:
            return

        self.ids['message'].text =  GRC_RES['OnCutting']
        rect = [int(item) for item in self.rect]
        img = self.srcimg.copy()
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        if (self.frame_or_mask == 0): 
            self.mask = np.zeros(self.srcimg.shape[:2],np.uint8)  # for mask initialized to PR_BG
            cv2.grabCut(img,self.mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
            self.frame_or_mask = 1
        elif (self.frame_or_mask == 1):
            cv2.grabCut(img,self.mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask==1) + (self.mask==3),255,0).astype('uint8')
        if self.ids['rdreform'].state == 'down':
            mask2 = rd.RDreform(mask2) # デフォルトで平滑化　詳細は rdlib4.py参照
        else:
            mask2 = rd.getMajorWhiteArea(mask2) # 最大白領域のみ抽出
        img4 = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        img4[:,:,3] = (127*(mask2//255))+128
        img4 = rd.draw2(mask2,img4,thickness=1,color=(0,0,250,255))
        self.ids['outimg'].texture = cv2kvtexture(img4,force3=False)
        self.silhouette = mask2
        self.ids['message'].text =  GRC_RES['Finished']

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
    
    orig = srcimg.copy() # オリジナルを複製
    rorig = srcimg.copy() # 回転用のオリジナルの複製
    img = orig.copy() # 操作用画像
    # 操作用画像
    img, imgbackup,halfimg,output,halfoutput,mask= prepareimg(img,limitsize = MAXIMAGESIZE)

    mouseCallBacker = myMouse('input')

    do_keyEventLoop()

# アプリケーションメイン 
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