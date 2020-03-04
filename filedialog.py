from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
 
import os

# import japanize_kivy
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
 
from kivy.core.text import LabelBase, DEFAULT_FONT 
# LabelBase.register(DEFAULT_FONT, "ipaexg.ttf") 
 
Builder.load_string('''
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
                on_release: root.load(filechooser.selection)
 
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
