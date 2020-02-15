from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
 
import os

import japanize_kivy
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
 
from kivy.core.text import LabelBase, DEFAULT_FONT 
# LabelBase.register(DEFAULT_FONT, "ipaexg.ttf") 
 
Builder.load_string('''
#: set dummy './prof.jpg'     # dummy = './prof.jpg' という意味
<Root>:                       # アプリケーションのウィンドウ構成の定義
    image: pic0               # Root クラスのメンバ変数 image は下に出てくる id pic0 を指すものとする。
    BoxLayout:     　　　　　　#  2つのボタンを横に並並べる。
        orientation: 'vertical' # 縦並び
        Image:                # 画像エリア
            id: pic0          # 画像の id を pic0 と名付ける
            source: dummy     # 画像のソースファイルのパス
        BoxLayout:
            size_hint_y: None
            height: 30
            Button:                                                # Load ボタン
                text: 'Load'
                on_release: root.show_load()
            Button:
                text: 'Save'                                      # Save ボタン
                on_release: root.show_save()
 
<LoadDialog>:                             # Load の際のポップアップウィンドウの定義
    BoxLayout:
        pos: root.pos
        orientation: "vertical"
        FileChooserListView:             # リストビューの FileChooser
            filters: ['*.png','*.jpg']      # 表示するファイルを .png, .jpg に限定
            path : '.'                           # 表示するフォルダはカレントフォルダ
            id: filechooser
        BoxLayout:
            size_hint_y: None
            height: 30
            Button
                text: "Cancel"                           # Cancel ボタンを押すと、
                on_release: root.cancel()           # cancel メソッドを実行
            Button:
                text: "Load"                              # Load ボタンを押すと、load メソッドを実行
                on_release: root.load(filechooser.selection[0])
 
<SaveDialog>:                                # Save の際のポップアップウィンドウの定義
    BoxLayout:
        pos: root.pos
        orientation: "vertical"
        FileChooserListView:
            id: filechooser
            filters: ['*.png','*.jpg']
            path : '.' # ファイルを選ぶと、そのファイル名をfilename フィールドに書き込む
            on_selection: filename.text = self.selection and self.selection[0] or ''
        TextInput:
            id: filename                # テキスト入力フィールドの id 
            size_hint_y: None
            height: 30
            multiline: False            # 複数行禁止
        BoxLayout:
            size_hint_y: None
            height: 30
            Button:
                text: "Cancel"              # Cancel ボタン
                on_release: root.cancel()   # cancel メソッドを実行
            Button:
                text: "Save"                # Save ボタンを押すと save を実行
                on_release: root.save(filechooser.path, filename.text)
''')
 
class LoadDialog(FloatLayout):
    load = ObjectProperty(None)     # load と cancel は実装なしで
    cancel = ObjectProperty(None)   # パラメータとして宣言だけしておく
 
class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)
 
class Root(FloatLayout):            # アプリケーションのメインウィンドウ
    image = ObjectProperty(None)    # 上の Root 定義内の image を指す
 
    def dismiss_popup(self):        # popup を閉じる
        self._popup.dismiss()
        Window.size = self.keepsize  # ウィンドウサイズを戻す
 
    def show_load(self):       # ファイル読み込みメソッド
        self.keepsize = Window.size  # 元のウィンドウサイズを記録 
        Window.size = (600,600)      # 一時的に 600x600 に広げる（狭める）
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9,0.9))    # 後述する
        self._popup.open()           # ポップアップを広げる
 
    def show_save(self):       # ファイル書き込みメソッド
        self.keepsize = Window.size
        Window.size = (600,600)        
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
 
    def load(self, filepath):   # ファイル読み込み時の動作
        self.image.source = filepath
        # self.ids['pic0'].source = filepath   こう書いても同じ意味になる
        self.dismiss_popup()
 
    def save(self, path, filename):   # ファイル書き込み時の動作
        print("path=",os.path.join(path, filename))
        self.dismiss_popup()
 
class TestApp(App):             # アプリケーションの定義
    def build(self):             # ウィンドウの構成
        Window.size = (300,300)
        return Root()
 
if __name__ == '__main__':
    TestApp().run()