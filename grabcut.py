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


dummy = rd.loadPkl('Primrose.pkl')
print(dummy.shape)

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
            text: './primrose.png'
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
                Image:
                    source: "./grabcut.png"
                    center_x: self.parent.center_x
                    center_y: self.parent.center_y
                    allow_stretch: False
            Button:
                id: plus
                text: "+"
                font_size: 30
            Button:
                id: minus
                text: "-"
                font_size: 40
''')
 
dummy = cv2.imread('.'+os.sep+'grabcut.png',-1)
print(dummy.shape)
# Window.size = (dummy.shape[1]*2+10,dummy.shape[0]+80)
 
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

class CV2Canvas(threading.Thread):

        def __init__(self,daemon=True):
            super(CV2Canvas,self).__init__()
            self.srcimg = rd.loadPkl('Primrose.pkl')
            self.gryimg = self.makegray()
            self.outimg = self.srcimg.copy()
            self.canvas = self.makeCVcanvas(self.srcimg,self.outimg)
            cv2.namedWindow("CVCanvas",cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE)
            cv2.imshow("CVCanvas",self.canvas)
            cv2.moveWindow("CVCanvas", 100, 100)
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
            pass

        # ３または４チャネル画像をdsグレイ化
        def makegray(self):
            img = self.srcimg
            if len(img.shape) == 3 :
                if img.shape[2] == 4: # Alpha チャネル付き
                    gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
                else:
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            return gray

from kivy.graphics.texture import Texture
class MyWidget(BoxLayout):
    mode = 'None'
    res = GRC_RES
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
 
class MyApp(App):
    def build(self):
        Window.size = (300,200)
        Window.Pos = (0,0)
        mywidget = MyWidget(self)
        mywidget.ids['sp0'].values = ('Open','Save','Quit')
        self.title = u'GrabCut'
        return mywidget

MyApp().run()


