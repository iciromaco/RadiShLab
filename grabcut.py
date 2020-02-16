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
 
Builder.load_string('''
<MyWidget>
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 1
            height: 32
            Spinner:
                size_hint_x: None
                width:100
                id: sp0
                text: 'File'
                on_text: root.do_menu()
            Label:
                id: path0
                text: './primrose.png'
        BoxLayout:
            orientation: 'vertical'
            size_hint_y: 8
            # ラベル
            Label:
                id: label1
                text_size: self.size
                font_size: 20
                text: 'File メニューで画像を開いてください'
                halign: 'center'
                valign: 'center'
            BoxLayout:
                Button:
                    id: allclear
                    text: "AC"
                    font_size: 25
                    size_hint: (.25, 1)
                Button:
                    id: eraser
                    text: "Eraser"
                    font_size: 25
                    size_hint: (.25, 1)
                Button:
                    id: framing
                    text: "Framing"
                    font_size: 25
                    size_hint: (.5, 1)
            BoxLayout:
                Button:
                    id: background0
                    text: "0 - Sure Background"
                    font_size: 20
                Button:
                    id: foreground1
                    text: "1 - Sure Foreground"
                    font_size: 20
            BoxLayout:
                Button:
                    id: background2
                    text: "2 - Possibly Background"
                    font_size: 20
                Button:
                    id: foreground3
                    text: "3 - Possibly Foreground"
                    font_size: 20
            BoxLayout:
                Button:
                    id: number0
                    text: "0"
                    font_size: 25
                Button:
                    id: number1
                    text: "1"
                    font_size: 25
                Button:
                    id: number2
                    text: "2"
                    font_size: 25
                Button:
                    id: number3
                    text: "3"
                    font_size: 25
            BoxLayout:
                Button:
                    id: rot90
                    text: "Rot+90"
                    font_size: 25
                Button:
                    id: rot270
                    text: "Rot-90"
                    font_size: 25
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
 

class CanvasThread(threading.Thread):
    def __init__(self, key, srcimg):
        super(CanvasThread,self).__init__()
        self.daemon = True
        self.key = key
        self.srcimg = srcimg
        self.gryimg = self.makegray()
        self.outimg = self.srcimg.copy()

    def loadimage(self,filepath):
        print('SD')
        self.srcimg = cv2.imread(filepath)
        self.outimg = self.srcimg.copy()
        self.gryimg = self.makegray()        
        print(filepath)

 
    def run(self):
        cv2.imshow("Input Image",self.srcimg)
        cv2.moveWindow("Input Image", 400, 400)
        cv2.imshow("GrabCut Image",self.outimg)
        cv2.moveWindow("GrabCut Image", 400+self.srcimg.shape[1], 400)
        key = cv2.waitKey(1)
        while key != 27: # ESC code = 27
            cv2.imshow("Input Image",self.srcimg)
            cv2.imshow("GrabCut Image",self.outimg)
            (x1,y1,w,h)=cv2.getWindowImageRect("Input Image")
            cv2.moveWindow("GrabCut Image",x1+w,y1-72)
            key = cv2.waitKey(1)
        cv2.destroyWindow("Input Image")

        # ３または４チャネル画像をdsグレイ化
    def makegray(self):
        img = self.srcimg
        if len(img.shape) == 3 :
            if img.shape[2] == 4: # Alpha チャネル付き
                img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
            else:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img

from kivy.graphics.texture import Texture
class MyWidget(BoxLayout):
    mode = 'None'
    def __init__(self,**kwargs):
        super(MyWidget,self).__init__(**kwargs)
        self.srcimg = rd.loadPkl('Primrose.pkl')
        self.key = -1
        self.canvasthread = CanvasThread(self.key,self.srcimg)
        self.canvasthread.start()
 
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
            self.canvasthread.key = 27

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
        print('SD-A')
        self.canvasthread.loadimage(filepath)
        print('SD-B')
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
        Window.size = (450,450)
        Window.Pos = (0,0)
        mywidget = MyWidget()
        mywidget.ids['sp0'].values = ('Open','Save','Quit')
        self.title = u'GrabCut'

        return mywidget
    
MyApp().run()


