# -*- coding: utf-8 -*-

from random import random
from kivy.app import App
from kivy.config import Config

# 起動時の解像度の設定
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '768')  # 16:9
Config.set('graphics', 'resizable', False)  # ウインドウリサイズ禁止

from kivy.lang import Builder 

from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import ObjectProperty
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.togglebutton import ToggleButton

from kivy.utils import get_color_from_hex   # 色の16進数表示を可能にする
from kivy.core.window import Window

Builder.load_string('''
#:import hex_color kivy.utils.get_color_from_hex

<ColorButton>:
    background_normal: 'color_button_normal.png'
    background_down: 'color_button_down.png'
    group: 'color'
    border: (5, 5, 5, 5)
    on_release: app.painter.ids['paint_area'].set_color(self.background_color)
    #on_release: app.painter.paint_id.set_color(self.background_color) # この方法でもset_corlorにアクセス可能
<MyCanvasWidget>:
    paint_id:paint_area
    id: canvas_area

    test:button1

    BoxLayout:
        orientation: 'vertical'
        height: root.height

        width: root.width
        MyPaintWidget:
            id: paint_area
            size_hint_y: 0.8

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.1
            Label:
                size_hint_x: 0.1
                text: 'Line width %s' % int(s1.value) if s1.value else 'Line width not set'
                color: 0,0,0,1
            Slider:
                id: s1
                size_hint_x: 0.9
                value: 3
                range: (1,100)
                step: 1
                on_touch_down:app.painter.ids['paint_area'].set_line_width(self.value)

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.1
            clear_btn:button1
            Button:
                id: button1
                text: "Clear"
                ont_size: 30
                on_release: app.clear_canvas()


            ColorButton:
                text: "white "
                color: 0, 0, 0 , 1
                background_color: hex_color('#ffffff')

            ColorButton:
                text: "black "
                state: 'down'
                background_color: hex_color('#000000')

            ColorButton:
                text: "red "
                background_color: hex_color('#ff0000')

            ColorButton:
                text: "biue "
                background_color: hex_color('#0000ff')

            ColorButton:
                text: "green "
                background_color: hex_color('#008000')

            ColorButton:
                text: "orange"
                background_color: hex_color('#ff4500')

            ColorButton:
                text: "purple"
                background_color: hex_color('#800080')
    Button:
        text: 'save'
        color: 1, 1, 1 , 1
        font_size: 20
        on_release: app.save_canvas()
        border: (2, 2, 2, 2)
        x: 0
        top: root.top
        width: 80
        height: 40

''')
class MyPaintWidget(Widget):
    #pass
    last_color = '' # 画面クリアを押された場合の最後の色
    line_width = 3  # 線の太さ

    def on_touch_down(self, touch):
        if Widget.on_touch_down(self, touch):
            return


        color = (random(), 1, 1)
        with self.canvas:
            #Color(*color, mode='hsv')
            #d = 30.
            #Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=self.line_width)

    def set_line_width(self, line_width=3):
        self.line_width = line_width

    def on_touch_move(self, touch):
        if touch.ud:    # スライダーを動かす際のエラーを解除するため
            touch.ud['line'].points += [touch.x, touch.y]

    def set_color(self, new_color):
        ''' 塗る色を変更する '''
        self.last_color = new_color
        self.canvas.add(Color(*new_color))


class MyCanvasWidget(Widget):

    def clear_canvas(self):
        MyPaintWidget.clear_canvas(self)


class MyPaintApp(App):
    #paint_id = ObjectProperty(None)
    #self.painter.test # これでClearボタンにアクセス可能

    def __init__(self, **kwargs):
        super(MyPaintApp, self).__init__(**kwargs)
        self.title = '画像表示'

    def build(self):
        parent = Widget()
        self.painter = MyCanvasWidget()

        # 起動時の色の設定を行う
        self.painter.ids['paint_area'].set_color(
            get_color_from_hex('#000000'))  #黒色を設定


        #clearbtn = Button(text='Clear')
        #clearbtn.bind(on_release=self.clear_canvas)

        #parent.add_widget(self.painter)

        #parent.add_widget(clearbtn)

        #return parent
        return self.painter

    #def clear_canvas(self, obj):
    def clear_canvas(self):
        '''
        画面をきれいにする。行うことは以下の2点
        1:画面をクリアーにする
        2:最後にセットしていた色をセットしなおす
        ''' 
        self.painter.ids['paint_area'].canvas.clear()
        self.painter.ids['paint_area'].set_color(self.painter.ids['paint_area'].last_color)

    def save_canvas(self):
        # 時間があるときに一時的にcanvas.beforeに背景を塗り潰す処理を加えるの
        # https://kivy.org/docs/api-kivy.core.window.html?highlight=screenshot#kivy.core.window.WindowBase.screenshot
        Window.screenshot();    # スクリーンショットを保存する
        #self.painter.export_to_png('a.png')    # 画像を保存する　ただしこのやり方だとウィンドウカラーが適用されないので描いていない部分が透明になる

class ColorButton(ToggleButton):
    def _do_press(self):
        '''
        何も押されていない状態で設定が解除されるのを防ぐためToggleButtonの関数を継承して変更する 
        Source code for kivy.uix.behaviors.button
        https://kivy.org/docs/_modules/kivy/uix/behaviors/button.html
        '''

        if self.state == 'normal':
            # ボタンを押されてない場合は状態を変更する
            ToggleButtonBehavior._do_press(self)

if __name__ == '__main__':
    Window.clearcolor = get_color_from_hex('#ffffff')   # ウィンドウの色を白色に変更する
    MyPaintApp().run()