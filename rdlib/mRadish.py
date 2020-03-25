'''
Interactive segmentation tool with mRadish

Usage:
   python -m mRadish

'''
import cv2
import numpy as np
import os, sys
from PIL import ImageGrab
 
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
from kivy.graphics import Color, Rectangle, Line, Ellipse, Point, GraphicException

from sympy import diff,symbols,solve,Abs,im,re

sys.path.append('./rdlib')
import pprint
pprint.pprint(sys.path)
import filedialog # self made library
import rdlib4 as rd # self made library

# Prohibit red circle displayed by right click
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,disable_multitouch') 

GRC_RES ={
'OpenImage':'File メニューで画像を開いてください',
'TopTip':'上端の位置を調整し、OKなら Enter キーを押してください', # State 0
'BottomTip':'下端の位置を調整し、OKなら Enter キーを押してください', # State 1
'LeftSide':'左側計算中。表示されたら Enter を押してください', # State 2
'RightSide':'右側計算中。表示されたら Enter を押してください', # State 3
'AxisCalc':'中心軸計算中。表示されたら Enter を押してください', # State 4
'Reform':'補正形状計算中。しばらくお待ちください。'  # State 5
}

FILEMENUS = {'Open':"開く",'Save':"保存",'ScreenShot':"スクリーンショット",'Quit':"終了"}

DUMMYIMGBIN = 'res/nezuko.pkl'
BUTTONH = 32
check = rd.loadPkl(DUMMYIMGBIN)
DUMMYIMG = cv2.cvtColor(rd.loadPkl(DUMMYIMGBIN), cv2.COLOR_GRAY2BGR)
picdic = rd.loadPkl("/res/picdic.pkl")
SAVEDIR = './results' # 'SaveDirFixed:True' の場合の保存場所  

BUTTONH = 64 # hight of menu and buttons
PENSIZE = 3 # size of drawing pen
UNITSIZE =  256 # 高さがこのサイズを超える画像は強制的にこのサイズにリサイズ
MINWW = 640 # ウィンドウ幅の下限　画像幅がこの値の半分以下の場合は幅マージンを加えます
MM = 5 # 枠指定した際、枠から MM ピクセルは背景としてマスクを作る　よって枠指定するときは対象からMMピクセルは離すこと

# Conversion from hexadecimal color representation to floating vector representation
from kivy.utils import get_color_from_hex
C2 = [
    get_color_from_hex('#FF00FF10'), # MAGENDA for BG
    get_color_from_hex('#FFFFFF10'), # WHITE for FG
]
COLORS = []
for i in range(len(C2)):
    cc = C2[i]
    item = 'Color({},{},{},mode="rgba",group="{}")'.format(cc[0],cc[1],cc[2],cc[3],str(i))
    COLORS.append(item)

MAGENTA = [255,0,255]    # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : MAGENTA, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_COLORS = [DRAW_BG,DRAW_FG]

CON_THICK = 1 # 輪郭線描画線の太さ
CON_COLOR = (0,255,0,255) # 輪郭線描画色

# デザイン定義
Builder.load_string('''
#:set BH 32
#:set dummypath './res/testpics/demoimg.jpg'
<mRadishConsole>:
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
                    on_text: root.do_filemenu()
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

# 計算不能箇所のデータの補間
def interporation(plist):
    while np.sum(plist) == np.inf: # np.inf を含むなら除去を繰り返す
        for i in range(len(plist)):
            if np.sum(plist[i]) == np.inf :
                if (i !=0 and i !=len(plist)-1) and np.sum(plist[i-1]+plist[i+1]) != np.inf: # 当該は無限で、前後は無限ではない場合
                    plist = np.r_[plist[0:i],[[int(round(((plist[i-1]+plist[i+1])/2)[0])),
                                              int(round(((plist[i-1]+plist[i+1])/2)[1]))]],plist[i+1:]]
                elif len(plist[i:])>=3 and np.sum(plist[i+1]+plist[i+2]) != np.inf:
                    plist = np.r_[plist[0:i],[plist[i+2]-2*(plist[i+2]-plist[i+1])],plist[i+1:]]
                elif len(plist[0:i])>=2 and np.sum(plist[i-1]+plist[i-2]) != np.inf:
                    plist = np.r_[plist[0:i],[plist[i-2]-2*(plist[i-2]-plist[i-1])],plist[i+1:]]
    return plist

# 補正形状を求める
def reformRadish(img,N=8,fl=None,fr=None,fc=None,n_samples=32):

    lpoints2,rpoints2,cpoints,dpc = NormalLadder30(img,fl,fr,fc,n_samples=n_samples)

    # 各サンプル点における幅を求める
    lpoints2 = interporation(lpoints2)
    rpoints2 = interporation(rpoints2)
    width = []
    for [lx,ly],[rx,ry] in zip(lpoints2,rpoints2):
        w = np.sqrt((lx-rx)*(lx-rx)+(ly-ry)*(ly-ry))
        width.append(w)

    # 各サンプル点までの距離を求める
    dp,lengths = rd.getDenseParameters(fc,n_samples=128,span=0,needlength=True)
    found = [0.0]
    i = 1
    for n in range(1,len(dpc)-1):
        t = dpc[n]
        while dp[i] < t:
            i = i+1
        if t == dp[i]:
            found.append(lengths[i])
        else:
            found.append(((dp[i]-t)*lengths[i-1]+(t-dp[i-1])*lengths[i])/(dp[i]-dp[i-1]))
    found.append(lengths[-1])

    width = [0]+width+[0] # 少し姑息だが　端点を閉じる目的　
    found = [found[0]]+found+[found[-1]]
        
    # 形状補正データのサンプルの生成
    samples = np.array([[int(round(l)),int(round(w))] for l,w in zip(width,found)])
    
    bez = rd.BezierCurve(N=N,samples=samples) # インスタンス生成
    cps,fc = bez.fit0()
    # 結果の描画
    return cps,fc

# 図的解法で求めた垂線による断面を描画するプログラム
def NormalLadder30(img,fl,fr,fc,n_samples=32):
    t = symbols('t')
    dpl = rd.getDenseParameters(fl,n_samples=n_samples,span=0) #  均等間隔になるようなパラメータセットを求める
    dpr = rd.getDenseParameters(fr,n_samples=n_samples,span=0) #  均等間隔になるようなパラメータセットを求める
    spl = lpoints = np.array([[int(float(fl[0].subs(t,s))),int(float(fl[1].subs(t,s)))] for s in dpl])
    spr = rpoints = np.array([[int(float(fr[0].subs(t,s))),int(float(fr[1].subs(t,s)))] for s in dpr])
    
    fcx,fcy = fc
    cpoints = [] # 左右の対応点を結ぶ線分と中心線の交点
    dpc = [] # その点のパラメータ
    for [xl,yl],[xr,yr] in zip(spl,spr):
        print('.',end='')
        ans = solve((xr-fcx)*(fcy-yl)-(fcx-xl)*(yr-fcy),t) # 左右の等間隔点を結ぶ線分と中心線の交点
        ansR = [re(i) for i in ans if float(Abs(im(i)))<0.00000001] # 解の実部
        sc = [i for i in ansR if i<=1.02 and -0.02<=i] # ０から１までの範囲の解を抽出 
        cpoints.append([int(float(fcx.subs(t,sc[0]))),int(float(fcy.subs(t,sc[0])))] if sc !=[] else [np.inf,np.inf])
        dpc.append(sc[0] if sc !=[] else np.inf)
    if dpc[0] == np.inf:
        dpc[0] = 0
        cpoints[0] = [int(float(fc[0].subs(t,0))),int(float(fc[1].subs(t,0)))]
    if dpc[-1] == np.inf:
        dpc[-1] = 1
        cpoints[-1] = [int(float(fc[0].subs(t,1))),int(float(fc[1].subs(t,1)))]
    cpoints = np.array(cpoints)
    
    # 上端点における法線は両サイドと交差しないことが多いので計算せずに端点をそのまま採用
    lpoints2 = [[int(float(fl[0].subs(t,0))),int(float(fl[1].subs(t,0)))]]
    rpoints2 = [[int(float(fr[0].subs(t,0))),int(float(fr[1].subs(t,0)))]]
    for t0 in dpc[1:-1]:
        ldata,rdata = rd.crossPointsLRonImg(img,fc,t0) # 中心線 fcのパラメータ t0の点の法線と画像輪郭の交点を図的に求める
        lpoints2.append(ldata)
        rpoints2.append(rdata)
    # 下端点における端点も計算せずにそのまま採用
    lpoints2.append([int(float(fl[0].subs(t,1))),int(float(fl[1].subs(t,1)))])
    rpoints2.append([int(float(fr[0].subs(t,1))),int(float(fr[1].subs(t,1)))])
    lpoints2 = np.array(lpoints2)
    rpoints2 = np.array(rpoints2)
    return lpoints2,rpoints2,cpoints,dpc

# Main Widget
class mRadishConsole(BoxLayout):
    windowsize = DUMMYIMG.shape[1]*2, DUMMYIMG.shape[0]+2*BUTTONH # 初期ウィンドウサイズ
    pictexture = {key:cv2kvtexture(picdic[key],force3 = False) for key in picdic}
    touchud = [] # touch.ud の記憶場所
    margin = int(1.2*UNITSIZE) - UNITSIZE

    def __init__(self,**kwargs):
        super(mRadishConsole,self).__init__(**kwargs)
        self.fState = 0 # 枠指定の状態 0:初期、1:1点指定済み、2:指定完了
        self.canvasgroups = []
        self.conthick = CON_THICK
        self.tobesaved = 'orig'
        self.currentOpendir = self.currentSavedir = os.getcwd() # カレントオープンディレクトリ
        self.fixsavedir = True
        self.setsrcimg(DUMMYIMG)
        self.dialogflag = False # ファイルオープンダイアログを開いているというフラグ
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        Clock.schedule_interval(self.update, 0.01) # ウィンドウサイズの監視固定化

    # 画像上の座標(imgx,imgy) に対応する canvas 座標を返す
    def imgC2kivyC(self,imgx,imgy,offset=0):
        m = self.margin # canvas のマージン
        h,w = self.srcimg.shape[:2]
        kx = imgx + m + offset
        ky = BUTTONH + h - imgy - 1
        return kx,ky

    #  canvas 座標に対応する画像上の座標(imgx,imgy) 返す
    def kivyC2imgC(self,kx,ky):
        m = self.margin # canvas のマージン
        h,w = self.srcimg.shape[:2]
        imgx = kx - m
        imgy = BUTTONH + h - 1 - ky
        return imgx,imgy

    # 自動で上下端点を見つけると同時に輪郭情報を保存する
    def autoTips(self):
        img = self.srcimg
        con,topTip,bottomTip,_stops,_smbtms = rd.findTips(img,con=[],top=0.1,bottom=0.8,topCD=0.5,bottomCD=0.5,mode=2)
        self.con = con
        self.topTip = topTip
        self.bottomTip = bottomTip
        self.drawTips()       

    # 軸端マークを描く
    def drawTips(self):
        h,w = self.srcimg.shape[:2]
        m = self.margin
        [[tpx,tpy]] = self.con[self.topTip]
        [[btx,bty]] = self.con[self.bottomTip]
        ktpx,ktpy = self.imgC2kivyC(tpx,tpy)
        kbtx,kbty = self.imgC2kivyC(btx,bty)
        if self.fState == 0:
            tpx,tpy = ktpx,ktpy
        else:
            tpx,tpy = kbtx,kbty
        with self.canvas:
            self.canvas.remove_group("a")
            self.ud = [Color(0, 1, 0, mode='rgba'),
                Rectangle(pos=(tpx, BUTTONH), size=(1, h),group="a"), # クロスカーソル 縦
                Rectangle(pos=(0, tpy), size=(w+2*m, 1),group="a"), # クロスカーソル　横
                # Rectangle(pos=(ktpx+w+2*m, BUTTONH), size=(1, h),group="a"),
                Color(1, 0, 0, mode='rgba'),
                Line(circle=(ktpx, ktpy, 1),width=2,group="a"),
                Line(circle=(ktpx, ktpy, 8),group="a"),
                Line(circle=(kbtx, kbty, 1),width=2,group="a"),
                Line(circle=(kbtx, kbty, 8),group="a") ]

    # 入力画像をセット
    def setsrcimg(self,srcimg):
        saved = self.children[:]
        self.clear_widgets()
        self.canvas.clear()
        for widget in saved:
            self.add_widget(widget)

        # setfState : 0 枠指定前　3 
        if len(srcimg.shape) == 2: # もともとグレイ
            gryimg = srcimg.copy()
        else:
            gryimg = cv2.cvtColor(srcimg,cv2.COLOR_BGR2GRAY)  
        _ret,bwimg = cv2.threshold(gryimg,127,255,cv2.THRESH_BINARY) # 単純２値化
        bwimg = rd.tiltZeroImg(bwimg) # 傾き補正
        bwimg = rd.makeUnitImage(bwimg,mr=1.2,unitSize=UNITSIZE) # マージン20%でサイズを正規化
        _ret,bwimg = cv2.threshold(bwimg,127,255,cv2.THRESH_BINARY) # 2階調化
        self.srcimg = bwimg
        self.srctexture = cv2kvtexture(bwimg) # ソース画像のテクスチャを生成
        self.outimg = cv2.cvtColor(bwimg,cv2.COLOR_GRAY2BGR)//2
        self.outtexture = cv2kvtexture(self.outimg)
        self.resetAll()

    # 画像読み込み直後の状態まで戻す
    def resetAll(self):
        self.ids['srcimg'].texture = self.srctexture # ソース画像のテクスチャを表示
        self.ids['outimg'].texture = self.outtexture # 仮結果画像
        self.autoTips() # 先端位置を自動決定
        self.drawTips() # 先端位置を描画
        self.fState = 0 # フェーズを端点未決定にセット
        self.ids['message'].text =  GRC_RES['TopTip']  

    # キーボードイベント処理
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers, samples=32,N=8):
        h,w = self.srcimg.shape[:2]
        m = self.margin
        conn = len(self.con)
        f = self.imgC2kivyC

        self.update(0.001)

        if self.fState == 0:
            tip = self.topTip # 上端の輪郭番号
            [[tx,ty]] = self.con[tip] 
        elif self.fState == 1:
            tip = self.bottomTip # 下端の輪郭番号
            [[tx,ty]] = self.con[tip]


        if self.fState == 0:
            if keycode[1] == 'enter':
                self.fState = 1
                self.ids['message'].text =  GRC_RES['BottomTip'] 
                self.drawTips()
            else:
                if keycode[1] == 'right' or keycode[1] == 'up':
                    tip = (tip-1) % conn
                elif keycode[1] == 'left' or keycode[1] == 'down':
                    tip = (tip+1) % conn
                self.topTip = tip
                self.drawTips() # 描画
        elif self.fState == 1:
            if keycode[1] == 'enter':
                self.fState = 2
                self.ids['message'].text =  GRC_RES['LeftSide']

                con = rd.getContour(self.srcimg)
                rdcimg = np.zeros_like(self.srcimg)  # 描画キャンバスの準備
                cv2.drawContours(rdcimg,con,-1,255,thickness=1)
                [[dtopx,dtopy]] = self.con[self.topTip]
                [[dbtmx,dbtmy]] = self.con[self.bottomTip]
                conLeft,conRight = rd.getCntPairWithCntImg(rdcimg,dtopx,dtopy,dbtmx,dbtmy)    
                Left = self.Left = rd.getSamples(conLeft,N=samples,mode='Equidistant')
                Right = self.Right = rd.getSamples(conRight,N=samples,mode='Equidistant')
                bezL = rd.BezierCurve(N=N,samples=Left)
                cpsL,self.fL = bezL.fit0()
                fL = self.fL
                Lpoints = list(np.array([f(x,y,w+2*m) for [x,y] in cpsL]).flatten())
                with self.canvas:
                    Color(0.8, 0.62, 0.27, mode='rgba')
                    Line(bezier=(list(Lpoints)),width=2)
            else:
                if keycode[1] == 'right' or keycode[1] == 'up':
                    tip = (tip+1) % conn
                elif keycode[1] == 'left' or keycode[1] == 'down':
                    tip = (tip-1) % conn
                self.bottomTip = tip
                self.drawTips() # 描画
        elif self.fState == 2 and keycode[1] == 'enter':
            self.fState = 3
            self.ids['message'].text = GRC_RES['RightSide']
            Right = self.Right
            bezR = rd.BezierCurve(N=N,samples=Right)
            cpsR,self.fR = bezR.fit0()
            fR = self.fR
            Rpoints = list(np.array([f(x,y,w+2*m) for [x,y] in cpsR]).flatten())
            with self.canvas:
                Color(0.27, 0.8, 0.5, mode='rgba')
                Line(bezier=Rpoints,width=2)
        elif self.fState == 3 and keycode[1] == 'enter':
            self.fState = 4
            self.ids['message'].text =  GRC_RES['AxisCalc'] 
            fl = self.fL
            fr = self.fR
            fc = (fl+fr)/2
            dp = rd.getDenseParameters(fc,n_samples=samples,span=0)
            samples = [[int(float(fc[0].subs('t',s))),int(float(fc[1].subs('t',s)))] for s in dp]
            samples = np.array(samples)
            bezC = rd.BezierCurve(N=4,samples = samples,prefunc=fc)
            cps,newfc = bezC.fit0()
            self.fC = newfc
            Cpoints = list(np.array([f(x,y,w+2*m) for [x,y] in cps]).flatten())
            with self.canvas:
                Color(1, 0, 0, mode='rgba')
                Line(bezier=Cpoints,width=2)
            self.ids['message'].text =  GRC_RES['Reform'] 
        elif self.fState == 4 and keycode[1] == 'enter':
            self.fState = 5
            self.ids['message'].text =  GRC_RES['Reform'] 
            fl = self.fL
            fr = self.fR
            fc = self.fC
            cps,fc = reformRadish(self.srcimg,N=8,fl=fl,fr=fr,fc=fc,n_samples=samples)
            gx,gy,(x0,y0,_w,_h,a) = rd.getCoG(self.srcimg)
            FpointsL = list(np.array([f(-x/2,y+y0,w+m) for [x,y] in cps]).flatten())
            FpointsR = list(np.array([f(x/2,y+y0,w+m) for [x,y] in cps]).flatten())
            with self.canvas:
                Color(1, 1, 1, mode='rgba')
                Line(bezier=FpointsL,width=2)
                Line(bezier=FpointsR,width=2)
            self.fState = 6
            self.ids['message'].text =  GRC_RES['OpenImage'] 
        else:
            self.ids['message'].text =  GRC_RES['OpenImage'] 
        return 

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None


    # ウィンドウサイズを固定化
    def update(self, dt):
        if self.dialogflag == True:
            return
        h,w = self.srcimg.shape[:2]
        Window.size = (2*w+4*self.margin,h+2*BUTTONH)
        if self.fState == 0:
            self.ids['message'].text =  GRC_RES['TopTip'] 
        elif self.fState == 1:
            self.ids['message'].text =  GRC_RES['BottomTip']             
        elif self.fState == 2:
            self.ids['message'].text =  GRC_RES['LeftSide'] 
        elif self.fState == 3:
            self.ids['message'].text = GRC_RES['RightSide']
        elif self.fState == 4:
            self.ids['message'].text =  GRC_RES['AxisCalc'] 
        elif self.fState == 5:
            self.ids['message'].text =  GRC_RES['Reform'] 
        else:
            self.ids['message'].text =  GRC_RES['OpenImage'] 

    # メニュー処理
    def do_filemenu(self):
        if self.ids['sp0'].text == 'File':
            return
        else:
            mode = self.ids['sp0'].text
            self.ids['sp0'].text = 'File'
        if mode == FILEMENUS['Open']: 
            self.show_load()
        elif mode == FILEMENUS['Save']:
            if self.fState > 2:
                self.show_save()
        elif mode == FILEMENUS['ScreenShot']:
            grabimg = np.asarray(ImageGrab.grab())
            grabimg = cv2.cvtColor(grabimg,cv2.COLOR_RGB2BGR)
            self.setsrcimg(grabimg)
        elif mode == FILEMENUS['Quit']:
            sys.exit()

    def do_prefmenu(self):
        if self.ids['sp1'].text == 'Preferences':
            return
        else:
            mode = self.ids['sp1'].text
            self.ids['sp1'].text = 'Preferences'
        if mode == PREFMENUS['ToggleThickness']:
            self.conthick = 1 if self.conthick == 2 else 2
            PREFMENUS['ToggleThickness'] = "{}{}".format(PREFMENUS['ToggleThickness'][:-1],self.conthick)
        elif mode == PREFMENUS['ToggleSave']:
            self.tobesaved = 'orig' if self.tobesaved == 'crop' else 'crop'
            PREFMENUS['ToggleSave'] = PREFMENUS['ToggleSave'][:-4]+self.tobesaved             
        self.ids['sp1'].values = (PREFMENUS[i] for i in PREFMENUS.keys())

    # ファイルウィンドウをポップした状態からの復帰
    def dismiss_popup(self):
        self._popup.dismiss()
        self.dialogflag = False
        Window.size = self.keepsize
 
    # ファイルの選択と読み込み
    def show_load(self):
        def load(filepath):
            if len(filepath)>0:
                self.ids['path0'].text = filepath[0]
                srcimg = rd.imread(filepath[0])
                if len(srcimg.shape) == 2:
                    srcimg = cv2.cvtColor(srcimg, cv2.COLOR_GRAY2BGR)
                self.setsrcimg(srcimg)
                self.currentOpendir = os.path.dirname(filepath[0])
            self.dismiss_popup()

        self.keepsize = Window.size
        self.dialogflag = True
        Window.size = (600,600)
        content = Factory.LoadDialog(load=load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9,0.9))
        content.ids['filechooser'].path = self.currentOpendir
        self._popup.open()
        
    def show_save(self):
        def save(dir, filename):
            if filename !='':
                path = os.path.join(dir, filename) # パスを合成
                path1 = os.path.splitext(path) # パスを拡張子より前と拡張子に分解
                if not path1constran.lower() in ['.png','.jpg']:
                    path = path1[0]+'.png' # 強制的に png に変更
                oh,ow = self.origin.shape[:2]
                img = np.zeros((oh,ow),np.uint8)
                (x,y,w,h) = self.cropRect
                img[y:y+h,x:x+w] = self.silhouette
                if self.tobesaved == 'crop':
                    img = img[y:y+h,x:x+w] # シルエット画像
                    simg = (self.origin.copy())[y:y+h,x:x+w] # オリジナル画像
                    spath = path1[0]+"_Org.png"
                    rd.imwrite(spath,simg)
                rd.imwrite(path,img)
                print("Write Image {} (x:{},y;{}),(w:{},h:{}))".format(path,x,y,w,h))
                self.currentSavedir = os.path.dirname(path)
            self.dismiss_popup()

        self.keepsize = Window.size
        self.dialogflag = True
        Window.size = (600,600)
        content = Factory.SaveDialog(save=save, cancel=self.dismiss_popup)
        filename = os.path.basename(self.ids['path0'].text)
        savename = "Sil_"+os.path.splitext(filename)[0]+".png"
        content.ids['filename'].text = savename
        self._popup = Popup(title="Save file", content=content, 
                            size_hint=(0.9, 0.9))
        content.ids['filechooser'].path = self.currentSavedir
        self._popup.open()

# アプリケーションメイン 
class mRadish(App):
    title = 'Touchtracer'

    def build(self):
        mywidget = mRadishConsole()
        sp0 = (FILEMENUS[i] for i in FILEMENUS.keys())
        mywidget.ids['sp0'].values = sp0
  
        self.title = 'mRadish'
        self.icon = 'res/picdicpics/ic99_radishB.png'
        return mywidget

    def on_pause(self):
        return True
    
if __name__ == "__main__":
    mRadish().run()