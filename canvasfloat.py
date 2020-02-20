import japanize_kivy

from kivy.app import App
from kivy.lang import Builder 
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
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

IF_H = 32 #  ボタンやメニューの高さ

Builder.load_string('''
#:set BH 32
#:set WINW 854
#:set WINH 704
#:set IMGH 640
#:set IMGW 427

<CanvasFloatLayout>:
    id: rdcanvas
    size_hint: None,None
    # size: self.parent.size 
    size: WINW,WINH
    pos: 0,BH        
    BoxLayout:
        orientation: 'horizontal'
        size_hint: None,None
        size: WINW,WINH
        pos: 0,BH
        Image:
            id: srcimg
            size_hint: None,None
            size: IMGW,IMGH
            allow_stretch:True
        Image:
            id: outimg
            size_hint: None,None
            size: IMGW,IMGH
''')

class CanvasFloatLayout(FloatLayout):
    
    def __init__(self,**kwargs):
        super(CanvasFloatLayout,self).__init__(**kwargs)
        self.setsrcimg(DUMMYIMG)
        h,w = DUMMYIMG.shape[:2]
        self.boxLayout = BoxLayout(
            orientation = 'horizontal',
            size_hint = (None,None),
            size = (2*w,h),
            pos = (0,IF_H)
        )
        self.add_widget(self.boxLayout)

        self.srcKvimg = Image(
            size_hint = (None,None),
            size = (w,h)
        )
        self.boxLayout.add_widget(self.srcKvimg)

        self.outKvimg = Image(
            size_hint = (None,None),
            size = (w,h)
        )
        self.boxLayout.add_widget(self.outKvimg) 
        self.setsrcimg(DUMMYIMG)
        
    def setsrcimg(self,srcimg):
        self.srcCVimg = srcimg
        h,w = srcimg.shape[:2]
        self.srcKvimg.texture = cv2kvtexture(srcimg)
        self.gryCVimg = self.makegray()
        self.outKvimg.texture = cv2kvtexture(self.gryCVimg)

    def makegray(self):
        img = self.srcCVimg
        if len(img.shape) == 3 :
            if img.shape[2] == 4: # Alpha チャネル付き
                gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return gray

    def on_touch_down(self, touch):
        if Widget.on_touch_down(self, touch):
            return

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

    def on_touch_move(self,touch):
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
                lp = ud['lines'][-1].add_point # add_point関数 を lp と alias している
                for idx in range(0, len(points), 2):
                    lp(points[idx], points[idx + 1])
            except GraphicException:
                pass

        ud['label'].pos = touch.pos
        self.update_touch_label(ud['label'], touch)

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return    
        touch.ungrab(self)
        ud = touch.ud
        self.canvas.remove_group(ud['group'])
        self.remove_widget(ud['label'])
    
    def update_touch_label(self, label, touch):
        h,w = self.srcCVimg.shape[:2]
        y = (h + IF_H) - touch.y
        label.text = '(%d, %d)' % (touch.x, y)
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
