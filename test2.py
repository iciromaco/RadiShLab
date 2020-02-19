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
from kivy.graphics.texture import Texture
# LabelBase.register(DEFAULT_FONT, "ipaexg.ttf") 
from kivy.clock import Clock
import cv2
import numpy as np
import os, sys
import threading
 
import filedialog
import rdlib4 as rd

from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle, Point, GraphicException
from math import sqrt
import time

Window.size = (1024,544)

GRC_RES ={'GRC_TEXT':['File メニューで画像を開いてください']}
DUMMYPATH = './Primrose.png'
PRIMROSE = './res/Primrose.pkl'
MARGINHEIGHT = 32
dummy = rd.loadPkl(PRIMROSE)
picdic = rd.loadPkl('./res/picdic.pkl')

Builder.load_string('''
<MyWidget>:
    size_hint: None,None
    size: root.windowsize
    FloatLayout:
        size_hint: None,None
        size: self.parent.size[0],root.m_size
        BoxLayout:
            orientation: 'horizontal'
            pos: 0,root.size[1]-root.m_size
            Spinner:
                size_hint_x:0.2
                id: sp0
                text: 'File'
                on_text: root.do_menu()
            Label:
                id: message
                text: root.res['GRC_TEXT'][0]
                halign: 'center'
                valign: 'center'
    FloatLayout:
        size_hint: None,None
        size: self.parent.size[0],self.parent.size[1]-2*root.m_size
        Touchtracer:
            orientation: 'horizontal'
            size_hint: None,None
            size: self.parent.size
            pos: 0,root.m_size
            canvas:
                Color:
                    rgb: 0, 0, 0
                Rectangle:
                    size: self.size        
            Image:
                id: srcimg
                size: self.texture_size
                # allow_stretch:True
                # texture: root.srctexture
            Image:
                id: outimg
                size: self.texture_size
                # texture: root.outtexture
    FloatLayout:
        size_hint: None,None
        size: self.parent.size[0],root.m_size
        pos: 0,0
        BoxLayout:
            orientation: 'horizontal'
            Label:
                id: path0
                text: root.imgpath
                font_size: 12
                size_hint_x: 0.8
            BoxLayout:
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

class Touchtracer(BoxLayout):

    def on_touch_down(self, touch):
        print(touch)
        win = self.get_parent_window()
        ud = touch.ud
        ud['group'] = g = str(touch.uid)
        pointsize = 5
        ud['color'] = 0

        with self.canvas:
            Color(ud['color'], 1, 1, mode='hsv', group=g)
            ud['lines'] = [
                Rectangle(pos=(touch.x, 0), size=(1, win.height), group=g), # クロスカーソル 縦
                Rectangle(pos=(0, touch.y), size=(win.width, 1), group=g), # クロスカーソル　横
                Point(points=(touch.x, touch.y), source='res/picdicpics/particle.png',
                                       pointsize=pointsize, group=g)]

        ud['label'] = Label(size_hint=(None, None))
        self.update_touch_label(ud['label'], touch)
        self.add_widget(ud['label'])
        touch.grab(self)
        return True

    def on_touch_move(self, touch):
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

class TouchtracerApp(App):
    title = 'Touchtracer'

    def build(self):
        return Touchtracer()

    def on_pause(self):
        return True


if __name__ == '__main__':
    TouchtracerApp().run()