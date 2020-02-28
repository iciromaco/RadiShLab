

'''
Interactive segmentation tool with grabCut

Usage:
   python ./iGrabit.py

'''
import cv2
import numpy as np
import os, sys
 
import japanize_kivy  #  pip install japanize_kivy

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder 
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.factory import Factory
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Point, GraphicException

import filedialog # self made library
import rdlib4 as rd # self made library

# Prohibit red circle displayed by right click
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
DUMMYIMGBIN = '/res/paprika.pkl'
# DUMMYIMGBIN = '/res/Primrose.pkl'
BUTTONH = 32
DUMMYIMG = rd.loadPkl(DUMMYIMGBIN)
picdic = rd.loadPkl('./res/picdic.pkl')

BUTTONH = 32 # hight of menu and buttons
PENSIZE = 3 # size of drawing pen
MAXHIGHT = 640 # 高さがこのサイズを超える画像は強制的にこのサイズ以下になるよう1/2,1/4にリサイズします
MINWW = 800 # ウィンドウ幅の下限　画像幅がこの値の半分以下の場合は幅マージンを加えます
MM = 5 # 枠指定した際、枠から MM ピクセルは背景としてマスクを作る　よって枠指定するときは対象からMMピクセルは離すこと

# Conversion from hexadecimal color representation to floating vector representation
from kivy.utils import get_color_from_hex
C2 = [
    get_color_from_hex('#FF00FF10'), # MAGENDA for BG
    get_color_from_hex('#FFFFFF10') # WHITE for FG
]
COLORS = []
for i in range(2):
    cc = C2[i]
    item = 'Color({},{},{},mode="rgba",group="{}")'.format(cc[0],cc[1],cc[2],cc[3],str(i))
    COLORS.append(item)

MAGENTA = [255,0,255]    # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : MAGENTA, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_COLORS = [DRAW_BG,DRAW_FG]

CON_THICK = 2 # 輪郭線描画線の太さ
CON_COLOR = (100,255,200,255) # 輪郭線描画色

# デザイン定義
Builder.load_string('''
#:set BH 32
#:set dummypath './Primrose.png'
<GrabCutConsole>:
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
                Button:
                    id: eraser
                    on_press: root.undoDraw1()
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['eraser']
                ToggleButton:
                    id: framing
                    group: "mode"
                    state : "down"
                    on_press: root.startFraming()
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['frame']
                ToggleButton:
                    id: mark0 # Sure Background
                    group: "mode"
                    on_press: root.on_markup(0)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['zero']
                ToggleButton:
                    id: mark1 # Sure Foreground
                    group: "mode"
                    on_press: root.on_markup(1)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['one']
                Button:
                    id: rot90
                    on_press: root.rotateImage(90)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['rot90']
                Button:
                    id: rot270
                    on_press: root.rotateImage(270)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['rot270']
                Button:
                    id: plus
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
                    on_press: root.thicknessUpDown(-1)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['minus']
                ToggleButton:
                    id: rdreform # reform flag
                    text: "RF"
                    on_press: root.grabcut0()
                    group: "rdreform"
                    state: "down"
                Button:
                    id: grabcut
                    on_press: root.grabcut()
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['cut']
''')

# Returns a sequence of points [x1, y1, x2, y2, ...] other than both ends, 
# 　　obtained by dividing the line segment connecting two points by the interval step
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

# Convert opencv color image (numpy array) to kivy texture
# If you need alpha channel, set face3 = False
def cv2kvtexture(img, force3 = True):
    if len(img.shape) == 2:
        img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3 or force3:
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
    else : # with alpha
        img2 = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) # BGRA to RGBA
    img2 = cv2.flip(img2,0)   # flip upside down
    height = img2.shape[0]
    width = img2.shape[1]
    texture = Texture.create(size=(width,height))
    colorfmt = 'rgb' if img2.shape[2]==3 else 'rgba'
    texture.blit_buffer(img2.tostring(), colorfmt=colorfmt)
    return texture

# Main Widget
class GrabCutConsole(BoxLayout):
    windowsize = DUMMYIMG.shape[1]*2, DUMMYIMG.shape[0]+2*BUTTONH # 初期ウィンドウサイズ
    pictexture = {key:cv2kvtexture(picdic[key],force3 = False) for key in picdic}
    touchud = [] # touch.ud の記憶場所

    def __init__(self,**kwargs):
        super(GrabCutConsole,self).__init__(**kwargs)
        self.fState = 0 # 枠指定の状態 0:初期、1:1点指定済み、2:指定完了
        self.canvasgroups = []
        self.setsrcimg(DUMMYIMG,setfState=0)
        self.rect = [0,0,1,1] # 切り出し枠
        self.fp1 = [0,0] # 切り出し枠枠の1点目の座標
        self.pointsize = PENSIZE # ペンサイズ
        self.pensizeimage()
        self.ids['message'].text = GRC_RES['OpenImage']
        Clock.schedule_interval(self.update, 0.1) # ウィンドウサイズの監視固定化

    # サイズが大きすぎると画面表示できないので表面上リサイズ
    def sizefilter(self,srcimg):
        h,w = srcimg.shape[:2]
        height = min(MAXHIGHT,h)
        self.resizeratio = h/height
        if h != height:
            h = int(h/self.resizeratio)
            w = int(w/self.resizeratio)
        return cv2.resize(srcimg,(w,h))

    # 入力画像をセット
    def setsrcimg(self,srcimg,setfState=0):
        self.fState = setfState
        if setfState == 0: # オリジナルの保存
            self.origin = srcimg.copy()
            self.srcimg = self.sizefilter(srcimg) # 大きすぎる画像は強制的に縮小して処理する
        else:
            self.srcimg = srcimg.copy()
        self.srctexture = cv2kvtexture(self.srcimg)
        self.ids['srcimg'].texture = self.srctexture
        h,w = self.srcimg.shape[:2]
        self.margin = (MINWW - 2*w)//4 if 2*w < MINWW else 0  # ウィンドウ幅はある程度必要なので確保
        self.initAll(setfState=setfState)

    # モードボタンの初期化
    def resetMode(self):
        for tb in ['mark0','mark1','framing']:
            self.ids[tb].state = 'normal'

    # 画像読み込み後の初期化
    def initAll(self,setfState=0):
        srcimg = self.srcimg
        for ids in self.canvasgroups:
            self.canvas.remove_group(ids)  # 登録された描画情報を除去
        self.canvasgroups = []
        self.resetMode() # mode ボタンを全て非選択に
        gryimg = cv2.cvtColor(srcimg,cv2.COLOR_BGR2GRAY) # グレイ画像作成
        self.mask = np.zeros(srcimg.shape[:2],np.uint8) # GrabCut 用マスクの初期化
        self.maskStack = [self.mask.copy()] # マスクのバックアップ
        self.silhouette = smask = rd.getMajorWhiteArea(gryimg,binary=True) # シルエット画像作成
        img4 = cv2.cvtColor(srcimg,cv2.COLOR_BGR2BGRA) # アルファチャネル追加
        img4[:,:,3] = (127*(smask//255)) +128 # 黒領域の透明化
        self.outimg = rd.draw2(smask,img4,thickness=CON_THICK,color=CON_COLOR) # 輪郭描画
        self.ids['outimg'].texture = cv2kvtexture(self.outimg,force3=False) # 仮結果画像

        self.fState = setfState # 枠指定の状態初期化
        self.frame_or_mask = 0 # 0 -> mask は初期状態 1 -> セット済み
        if setfState == 0:
            self.ids['framing'].state = "down"
        elif setfState == 3:
            self.ids['mark0'].state = 'down'

    # 画像読み込み直後の状態まで戻す
    def resetAll(self):
        self.setsrcimg(self.origin.copy(),setfState=0)

    # ペンサイズの増減
    def pensizeimage(self):
        pensize = self.pointsize
        pimg = np.ones((32,32,4),np.uint8)*200
        pimg[:,:,3] = 100
        cv2.circle(pimg,(16,16),pensize,(255,255,255,255),-1)
        ptxt = cv2kvtexture(pimg,force3=False)
        self.ids['dotsize'].texture = ptxt

    # ウィンドウサイズを固定化
    def update(self, dt):
        h,w = self.srcimg.shape[:2]
        Window.size = (2*w+4*self.margin,h+2*BUTTONH)
        if self.fState == 0:
            self.ids['message'].text =  GRC_RES['TopLeft'] # メッセージ表示
        elif self.fState == 1:
            self.ids['message'].text =  GRC_RES['BottomRight'] # メッセージ表示            
        else:
            marking = self.nowMarking()
            if marking < 0:
                self.ids['message'].text =  GRC_RES['Finished'] # メッセージ表示
            else:
                self.ids['message'].text = GRC_RES['Marking%d' % (marking)]

    # 画像を９０度回転
    def rotateImage(self,rot=90):
        if self.fState >= 2:
            return
        img = self.origin
        if rot == 90:
            img = img.transpose(1,0,2)[::-1,:,:]
        else:
            img = img.transpose(1,0,2)[:,::-1,:]
        self.setsrcimg(img,setfState=0)

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
            if len(filepath)>0:
                self.ids['path0'].text = filepath[0]
                srcimg = rd.imread(filepath[0])
                self.setsrcimg(srcimg,setfState=0)
            self.dismiss_popup()

        self.keepsize = Window.size
        Window.size = (600,600)
        content = Factory.LoadDialog(load=load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9,0.9))
        self._popup.open()
        
    def show_save(self):
        def save(path, filename):
            if filename !='':
                path = os.path.join(path, filename)
                path1 = os.path.splitext(path)
                if not path1[1].lower() in ['.png','.jpg']:
                    path = path1[0]+'.png'
                self.ids['path0'].text = path
                oh,ow = self.origin.shape[:2]
                img = np.zeros((oh,ow),np.uint8)
                x,y,w,h = self.cropRect
                img[y:y+h,x:x+w] = self.silhouette
                rd.imwrite(path,img)
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
        m = self.margin
        l = y >= BUTTONH and y < h + BUTTONH 
        l = l and ((m <= x and x < w + m) or (w + 3*m <= x and x < 2*w + 3*m))
        return l

    # 枠付けの開始
    def startFraming(self):
        if self.fState >= 2:
            self.ids['framing'].state = "normal"
        else:
            self.ids['framing'].state = "down"

    # 枠付け中であるかどうかの判定
    def nowFraming(self):
        return self.fState < 2 and self.ids['framing'].state == "down" 

    # マーキングの切り替え
    def on_markup(self,ret):
        if self.fState < 2:
            self.ids['mark%d' % ret].state = "normal"
            self.ids['framing'].state = "down"
        else:
            if self.ids['mark%d' % ret].state == "down":
                self.ids['message'].text = GRC_RES['Marking%d' % (ret)]

    # マーキング中であるかどうかの判定
    def nowMarking(self):
        ret = -1
        for n in range(2):
            if self.ids['mark%d' % n].state == 'down':
                ret = n
        return ret

    # マーキング　ヒント情報の描画
    def drawPoint(self,points,colorvalue):
        m = self.margin
        for idx in range(0,len(points),2):
            x = int(points[idx])
            y = self.srcimg.shape[0]+BUTTONH-int(points[idx+1])
            # cv2.circle(self.workimg,(x,y),self.pointsize,colorvalue['color'],-1)
            cv2.circle(self.mask,(x-m,y),self.pointsize,colorvalue['val'],-1)

    # マーキングのアンドゥ
    def undoDraw1(self):
        lastid = self.popCV()
        if lastid == False:
            self.setsrcimg(self.origin,setfState=0)
            return
        elif lastid == "0":
            self.fState = 0
        elif lastid == "1":
            self.fState = 1
        else:
            self.fState = 3
        self.popMask()
        if self.fState < 2: 
            self.ids['framing'].state = "down"
            self.frame_or_mask = 0

    # ペンサイズの増減
    def thicknessUpDown(self, diff):
        pointsize = self.pointsize + diff
        if pointsize > 0 and pointsize < 31: 
            self.pointsize = pointsize
        self.pensizeimage()

    # 描画履歴情報のプッシュ・ポップ
    def pushCV(self,id):
        self.pushMask()  # マスクもプッシュ
        self.canvasgroups.append(id)

    def popCV(self):
        if len(self.canvasgroups) == 0:
            return False
        lastid = self.canvasgroups.pop()
        self.canvas.remove_group(lastid)
        return lastid  

    # マスク情報のプッシュ・ポップ
    def pushMask(self):
        self.maskStack.append(self.mask.copy())

    def popMask(self):
        if len(self.maskStack) == 1:
            self.mask = self.maskStack[0].copy()
        else:
            self.mask = self.maskStack.pop().copy()

    # マウスイベントの処理
    def on_touch_down(self, touch):
        if Widget.on_touch_down(self, touch) or not self.isInCanvas(touch): 
            # ボタンなどのウィジット上でタッチした場合は処理をスルー
            return
        
        h,w = self.srcimg.shape[:2]
        m = self.margin
        ud = touch.ud
        x = touch.x if touch.x < w + 2*m else touch.x - (w + 2*m)
        
        if self.nowFraming(): # 枠設定中
            g = str(self.fState) 
            with self.canvas:
                # Color(ud['color'], 1, 1, mode='hsv', group=g)
                Color(0, 1, 0, mode='rgba',group=g)
                ud['cross'] = [
                    Rectangle(pos=(x, BUTTONH), size=(1, h), group=g), # クロスカーソル 縦
                    Rectangle(pos=(0, touch.y), size=(2*(w+2*m), 1), group=g), # クロスカーソル　横
                    Rectangle(pos=(x+w+2*m, BUTTONH), size=(1, h), group=g)]
            ud['label'] = Label(size_hint=(None, None))
            self.update_touch_label(ud['label'], touch)
            self.add_widget(ud['label'])
        else:
            mark = self.nowMarking()
            if mark < 0: # not on marking
                return

            ud['group'] = g = str(touch.uid)
            ps = self.pointsize 

            with self.canvas:
                exec(COLORS[mark])
                # ud['drawings'] = Point(points=(touch.x, touch.y), source='res/picdicpics/particle.png',
                ud['drawings'] = [Point(points=(x, touch.y), source='res/picdicpics/pennib.png', 
                                      pointsize=ps, group=g),
                                 Point(points=(x+w+2*m, touch.y), source='res/picdicpics/pennib.png',
                                      pointsize=ps, group=g)] 
                self.drawPoint(ud['drawings'][0].points,colorvalue=DRAW_COLORS[mark])
        self.pushCV(g)
        touch.grab(self) # ドラッグの追跡を指定            
        return True

    def on_touch_move(self,touch):
        # 入力画像内でドラッグが開始されて、継続して画像内でドラッグが続いているかどうかのチェック
        if (touch.grab_current is not self) or (not self.isInCanvas(touch)):
            return

        h,w = self.srcimg.shape[:2]
        m = self.margin
        ud = touch.ud
        x = touch.x if touch.x < w + 2*m else touch.x - (w + 2*m)

        if self.nowFraming(): # 枠設定中
            ud['cross'][0].pos = x, BUTTONH
            ud['cross'][1].pos = 0, touch.y
            ud['cross'][2].pos = x + w + 2*m, BUTTONH                
            self.update_touch_label(ud['label'], touch)
        else:
            mark = self.nowMarking()
            if mark < 0:
                return

            ud['group'] = g = str(mark) 
            ps = self.pointsize

            while True:
                try:
                    pts = ud['drawings'][0].points
                    oldx, oldy = pts[-2], pts[-1]
                    break
                except:
                    index -= 1

            # カーソル移動が早すぎて間が飛んだ時のための補間処理
            points = calculate_points(oldx, oldy, x, touch.y, steps=ps)
            if points:
                try:
                    lp0 = ud['drawings'][0].add_point # add_point関数 を lp と alias している
                    lp1 = ud['drawings'][1].add_point # 
                    for idx in range(0, len(points), 2):
                        lp0(points[idx], points[idx + 1])
                        lp1(points[idx]+w+2*m, points[idx + 1])
                except GraphicException:
                    pass

                self.drawPoint(ud['drawings'][0].points,colorvalue=DRAW_COLORS[mark])
            
    def on_touch_up(self, touch):
        if touch.grab_current is not self or (not self.isInCanvas(touch)):
            return    

        h,w = self.srcimg.shape[:2]
        m = self.margin        
        ud = touch.ud

        if self.nowFraming(): # 枠設定中
            if self.fState == 0: # １点目未設定
                self.fp1[0] = int(ud['cross'][0].pos[0]-m)
                self.fp1[1] = int((h+BUTTONH)-ud['cross'][1].pos[1])
                # self.ids['message'].text =  GRC_RES['BottomRight']
                self.fState = 1 # 1点目確定
            elif self.fState == 1:
                p2x = int(ud['cross'][0].pos[0]-m)
                p2y = int((h+BUTTONH)-ud['cross'][1].pos[1])
                self.rect = [min(self.fp1[0],p2x),min(self.fp1[1],p2y),abs(self.fp1[0]-p2x),abs(self.fp1[1]-p2y)]
                self.ids['framing'].state = "normal"
                self.fState = 2
            self.remove_widget(ud['label'])
            self.grabcut()    
        else:
            touch.ungrab(self)

    # 座標表示
    def update_touch_label(self, label, touch):
        h,w = self.srcimg.shape[:2]
        m = self.margin
        label.text = '(%d, %d)' % (touch.x-m, (h + BUTTONH) - touch.y)
        label.texture_update()
        label.pos = touch.pos
        label.size = label.texture_size[0] + 20, label.texture_size[1] + 20

    # 平滑化モードボタンのふるまい
    def grabcut0(self):
        self.grabcut() # モードをトグルした上で grabcut

    # セグメンテーション
    def grabcut(self):
        if self.fState < 2:
            return
        elif self.fState == 2:
            [x,y,w,h] = self.rect
            rr = self.resizeratio
            self.cropRect = [int(rr*x),int(rr*y),int(rr*w),int(rr*h)]
            [x,y,w,h] = self.cropRect
            self.srcimg = self.origin[y:y+h,x:x+w]
            self.setsrcimg(self.srcimg,setfState=3)
            self.rect = [MM,MM,w-2*MM,h-2*MM]
        h,w = self.srcimg.shape[:2]
        self.ids['message'].text =  GRC_RES['OnCutting']
        rect = self.rect
        img = self.srcimg.copy()
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        if (self.frame_or_mask == 0): 
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
        img4[:,:,3] = (127*(mask2//255))+64
        img4 = rd.draw2(mask2,img4,thickness=CON_THICK,color=CON_COLOR)
        self.ids['outimg'].texture = cv2kvtexture(img4,force3=False)
        self.silhouette = mask2
        self.ids['message'].text =  GRC_RES['Finished']

# アプリケーションメイン 
class GrabCut(App):
    title = 'Touchtracer'

    def build(self):
        mywidget = GrabCutConsole()
        mywidget.ids['sp0'].values = ('Open','Save','Quit')
        self.title = 'GrabCut'
        return mywidget

    def on_pause(self):
        return True

if __name__ == '__main__':
   GrabCut().run()