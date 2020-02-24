#:set BH 32
#:set dummypath './Primrose.png'
BoxLayout:
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
                    on_press: root.undoDraw1()
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
                    on_press: root.rotateImage(90)
                    Image:
                        center_x: self.parent.center_x
                        center_y: self.parent.center_y
                        texture: root.pictexture['rot90']
                Button:
                    id: rot270
                    text: "R-"
                    on_press: root.rotateImage(270)
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
                        texture: root.pic