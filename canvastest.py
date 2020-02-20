from kivy.lang import Builder
from kivy.app import runTouchApp


root = Builder.load_string(r'''
FloatLayout:
    canvas.after:
        ScissorPush:
            size: int(self.width *0.7), int(self. height *0.7)
            pos: 0, 0
        Ellipse:
            size: self.size
            pos: self.pos
        ScissorPop:
''')

runTouchApp(root)