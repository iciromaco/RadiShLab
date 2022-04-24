from distutils.fancy_getopt import fancy_getopt
from pickletools import read_decimalnl_long
import PySimpleGUI as sg
import rdlib as rd
from rdlib.BezierCurve import BezierCurve 
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import io
import numpy as np
from sympy import solve,symbols,Abs,im, re,diff

DUMMYPATH = './res/testpics/demoimg.jpg'
DUMMYSHIL = './RDSamples/NSilImages/N_Sil_17daruma5o03_l.png'
DUMMYIMGBIN = 'res/sample.pkl'
DUMMYIMG = rd.loadPkl(DUMMYIMGBIN)
SRCTOP='.'

# Globals
'''
cv_inimg = {}  #  入力画像（cv2 BGR）
cv_inimgG = {} # 入力画像（cv2 Gray）
cv_outimg = {} # 出力画像 (cv2 BGR)
dp_inimg = {}  # 入力画像 (bytes形式)
dp_outimg = {} # 出力画像 (bytes形式) 
dpi_size = {'-tab0-':(480,480),'-tab1-':(320,320),'-tab2-':(320,320),'-tab3-':(320,320)} # 入力表示画像のサイズ
dpo_size = {'-tab0-':(480,480),'-tab1-':(600,640),'-tab2-':(600,640),'-tab3-':(600,600)} # 出力表示画像のサイズ
ioratio = {'-tab0-':1.0,'-tab1-':1.0,'-tab2-':1.0,'-tab3-':1.0}    # 入力画像サイズ/出力画像サイズ
'''
lasttab = '-tab0-'
'''
lastpath = {'-tab0-':DUMMYPATH,'-tab1-':DUMMYSHIL,'-tab2-':DUMMYSHIL,'-tab3-':DUMMYSHIL}
lastpathG = {'-tab0-':DUMMYPATH,'-tab1-':DUMMYSHIL,'-tab2-':DUMMYSHIL,'-tab3-':DUMMYSHIL}'''
#lastpathG = DUMMYSHIL
wholeicon = './res/picdicpics/Wholemini.png'
halvesicon = './res/picdicpics/Halvesmini.png'
Seg_order = 1
Seg_Maxoder = 3
funcL,funcR = None,None

# 指定したキャンバスに画像を表示
def drawImage(window,img_data=None,canvas='-I_CANVAS0-',tabn=0):
        if canvas == 0 or canvas == 'in':
            canvas = f'-I_CANVAS-{tabn}' 
        elif canvas == 1 or canvas == 'out':
            canvas = f'-O_CANVAS{tabn}-'
        window[canvas].erase()
        window[canvas].DrawImage(data=img_data, location=(0, 0))

# 動作確認用
def check(img,wname="TEST"):
    cv2.imshow(wname,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fitBezier(image=None,N=6,mode=0,Nsamples=24,err_th=1.0,savesize=(600,600)):
    # global dp_outimg,cv_inimgG,cv_outimg,funcL,funcR
    conLeft,conRight = rd.getCntPairWithImg(image,dtopdr=10,dbtmdr=10)
    SamplesL = rd.getSamples(conLeft,N=Nsamples,mode='Equidistant')
    SamplesR = rd.getSamples(conRight,N=Nsamples,mode='Equidistant') 
    bezL = BezierCurve(N=N,samples=SamplesL) # インスタンス生成
    bezR = BezierCurve(N=N,samples=SamplesR) # インスタンス生成
    moptionL = rd.getMutPoints(cont=conLeft,Samples=SamplesL)
    moptionR = rd.getMutPoints(cont=conRight,Samples=SamplesR)
    if mode == 0:
        bestcpsL, funcL, minerrorL = bezL.fit1T(maxTry=1000, mode=1, lr=0.015,lrP=1140,withErr=True, withEC=False,tpara=[], pat=300, err_th=err_th, threstune=1.0,moption=moptionL)
        bestcpsR, funcR, minerrorR = bezR.fit1T(maxTry=1000, mode=1, lr=0.015,lrP=1140,withErr=True, withEC=False,tpara=[], pat=300, err_th=err_th, threstune=1.0,moption=moptionR)
    else:
        bestcpsL, funcL= bezL.fit0()
        bestcpsR, funcR= bezR.fit0()
    # 描画
    item = io.BytesIO()
    rd.drawBez(image,stt=-0.00,end=1.00,bezL=funcL,bezR=funcR,cntL=SamplesL,cntR=SamplesR,saveImage=True,savepath=item,savesize=savesize)
    outimg = item.getvalue()
    return outimg,funcL,funcR

# ---------------------------------
# -----  Class Segmentation -------
# ---------------------------------
iconnames = {'-bAllClear-':[None,'AC'],'-TurnR90-':['ic07_90plus.png',''],
        '-TurnL90-':['ic06_90minus.png',''],'-bCut-':['ic09_cut.png',''],
        '-Frame-':['ic05_frame.png',''],'-mark0-':['ic00_zero.png',''],
        '-mark1-':['ic01_one.png',''],'-Eraser-':['ic08_eraser.png',''],
        '-NibsThin-':['ic11_minus.png',''],'-'
        '-NibsFat-':['ic10_plus.png',''],'-Smooth-':[None,'RF']}
MAXRsIZE = 640  # 入力画像の最大サイズ
icp = ['./res/picdicpics/'+ name[1][0] if name[1][0] else None for name in iconnames.items()]
icptext = [name[1][1] for name in iconnames.items()]
bkeys = list(iconnames.keys())

# -----  Segmentation 補助関数
# 入力画像のサイズが大きすぎると画面表示できないので強制リサイズする
def forceResizeImg(srcimg):
    h,w = srcimg.shape[:2]
    if max(h,w) > MAXRsIZE:
        if h > w:
            height = MAXRsIZE
            f_resizeratio = MAXRsIZE/h
            width = round(w*f_resizeratio)
        else:
            width = MAXRsIZE
            f_resizeratio = MAXRsIZE/w
            height = round(h*f_resizeratio)
        print('The image is too large. Forced to be %d x %d ' % (width,height))
    else:
        height,width,f_resizeratio = h,w,1.0
    resized = cv2.resize(srcimg,(width,height))
    return resized,f_resizeratio

# 表示用画像を生成  OpenCVイメージ　→　表示用ＴＫイメージ
def getDisplayImageWithRatio(cvimg,forcesize=(480,480)):
    tkimg, pilimg = rd.cv2tkimgwithPIL(cvimg,resize=forcesize)
    ioratio = cvimg.shape[0]/pilimg.height
    size = pilimg.size
    return tkimg, ioratio, size # tiimg:表示用画像（bytes型）、ioratio出力に対する入力の比率  size:表示画像の(幅,高さ)

###########################
class RLToolTab:
    dpi_size = (320,320)
    dpo_size = (600,600)
    path = None
    cv_inimg = None
    cv_inimg = None
    cv_outimg = None
    dp_inimg = None
    dp_outimg = None
    ioratio = 1.0
    Seg_order = 1
    Seg_Maxoder = 1

    def __init__(self,window,tab):
        self.window = window
        self.tab = tab

    # 入力画像のセット
    def setImage(self,path):
        if path == self.path:
            pass
        tab = self.tab
        window = self.window
        self.path = path
        if path == DUMMYPATH:
            self.cv_inimg = DUMMYIMG
        else:
            self.cv_inimg = cv2.imread(path,cv2.IMREAD_COLOR)
            self.cv_inimg, rr = forceResizeImg(self.cv_inimg)
        self.dp_inimg, self.ioratio, self.dpi_size = getDisplayImageWithRatio(self.cv_inimg)
        if window == None:
            window = sg.Window('Radish Lab Demo',make_layout(), resizable=True, finalize=True,size=(1200,680))
        drawImage(window,img_data=self.dp_inimg,canvas=f'-I_CANVAS{tab[4]}-')
        window[f'-O_CANVAS{tab[4]}-'].erase()
        window['-path-'].update(path)
        # window[tab].select()
        self.cv_inimgG = cv2.cvtColor(self.cv_inimg, cv2.COLOR_BGR2GRAY)  # グレイ化する
        if tab == '-tab0-':
            _ret, bwimg = cv2.threshold(self.cv_inimgG, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # ２値化
            wordermax, _, =  cv2.connectedComponents(bwimg) # 白領域の数
            self.Seg_order = 1
            self.Seg_Maxoder = min(3,wordermax) # 白領域の候補は最大３までとする
        return window

    # 画像サイズにあわせてウィンドウを作成
    @classmethod
    def make_double_canvas(cls):
        iwidth,iheight = cls.dpi_size
        owidth,oheight = cls.dpo_size
        tab = cls.tab
        input_canvas = sg.Graph(canvas_size=(iwidth,iheight),graph_bottom_left=(0,iheight),
                    graph_top_right=(iwidth,0),key=f'-I_CANVAS{tab[4]}-' , background_color='black',
                    enable_events=True,drag_submits=True,expand_x = False,expand_y = False) 
        output_canvas = sg.Graph(canvas_size=(owidth,oheight),graph_bottom_left=(0,oheight),
                    graph_top_right=(owidth,0),key=f'-O_CANVAS{tab[4]}-' , background_color='black',
                    enable_events=True,drag_submits=True,expand_x = False,expand_y = False) 
        return sg.Col([[sg.Col([[input_canvas],],vertical_alignment='top'),sg.VSeperator(),output_canvas]])

# --------------------------
# --- Class Segmentation ---
# --------------------------
class Seg(RLToolTab):
    Whole_or_HalfHalf = 'half'
    Auto_or_Manual = 'auto'
    eventlist = {'-EXTRACT-':'抽出実行','-ChangeT-':'ターゲット変更',
        '-TurnR90-':'90度右回転','-TurnL90-':'90度左回転',
        '-NibsThin-':'ペン先を小さく','-NibsFat-':'ペン先を太く',
        '-Smooth-':'抽出後平滑化','-Eraser-':'消しゴムモード',
        '-AllClear-':'マーキング全クリア','-Rewind-':'戻す',
        '-Frame-':'フレーミング','-Mark0-':'背景指定','-Mark1-':'前景指定'}
    tab = '-tab0-'
    dpi_size = (480,480)
    dpo_size = (480,480)

    def __init__(self,window):
        super().__init__(window, tab=self.tab)
        self.setImage(DUMMYPATH)
        
    def setImage(self,path):
        super().setImage(path)
        self.window['-SHRINK-'].update('0.80')
        self.Auto_or_Manual = 'auto'
        self.window['-ORDER-'].update(1)

    # レイアウトの初期化
    @classmethod
    def makelayout(cls):
        # select_layout = [[control_layout0]]
        control_layout0 = [
            [sg.Button(key = bkeys[i],image_filename=icp[i],button_text=icptext[i],button_color=('yellow','brown')) for i in range(4)],
            [sg.Button(key = bkeys[i],image_filename=icp[i],button_text=icptext[i],button_color=('yellow','brown')) for i in range(4,8)],    
            [sg.Button(key = bkeys[i],image_filename=icp[i],button_text=icptext[i],button_color=('yellow','brown')) for i in range(8,11)]+[sg.Graph((32,32),(0,32),(32,0),key='-pnib-')],    
        ]
        layout = [
            [sg.Col([[sg.T('シルエット抽出'),],
                [sg.Frame('Control',
                    [[sg.Text('Smoothing'),sg.Spin([f"{(0.5+i/20):.2f}" for i in range(10)],"0.80",readonly=False,key='-SHRINK-')],
                    [sg.Button('Extract',key = '-EXTRACT-',enable_events=True)],
                    [sg.Button('Change Target',key='-ChangeT-',enable_events=True),sg.Spin([1,2,3],1,readonly=False,key='-ORDER-')]
                    ])],
                [sg.Frame('Manual Segmentation',layout=control_layout0)]
            ],vertical_alignment='top'),
            Seg.make_double_canvas()]]
        return layout

    # AutoとManualの切り替え
    def toggleAM(self):
        self.Auto_or_Manual = 'auto' if self.Auto_or_Manual == 'manual' else 'manual'

    # 全自動セグメンテーション
    def autoSegmentation(self,order=1,shrink=0.8):
        # global cv_outimg,dp_outimg
        self.cv_outimg = self.getSilhouette(self.cv_inimgG,order=order,shrink=shrink) # シルエット生成
        self.dp_outimg = rd.cv2tkimg(self.cv_outimg,resize=self.dpo_size) # 出力表示用シルエット画像

    # シルエット画像生成
    def getSilhouette(seg,img,order=1,shrink=0.8):
        # order 何番目の白領域を取り出すか
        # shrink < 1   平滑化の程度　小さいほど平滑化効果が大きい
        if len(img.shape) == 2: # 白黒画像の場合はカラー化
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        _,oimg = rd.getRadish(img,order=order, shrink=shrink)
        return oimg 

    # イベント処理
    def Loop(self,event,values):
        if event == '-ChangeT-':
            order = values['-ORDER-']
            order = order + 1
            if order > Seg_Maxoder:
                order = 1
            values['-ORDER-']=order
            window['-ORDER-'].update(order)
            event = '-EXTRACT-'
        if event == '-EXTRACT-':
            order = values['-ORDER-']
            shrink = float(values['-SHRINK-'])
            self.autoSegmentation(order=order,shrink=shrink) # 自動セグメンテーションを実行
            drawImage(self.window,img_data=self.dp_outimg,canvas=f'-O_CANVAS0-')
                        

# ---------------------------------
# --- Class Fourier Descriptors ---
# ---------------------------------
# フーリエ記述子
F_menu = ['description', ['楕円フーリエ記述子', 'Z型フーリエ記述子', 'G型フーリエ記述子', 'P型フーリエ記述子']]
class Fur(RLToolTab):
    eventlist = {'-FDMenu-':'フーリエ記述子'}
    tab = '-tab1-'

    def __init__(self,window):
        super().__init__(window, tab='-tab1-')
        self.setImage(DUMMYSHIL)
        self.Whole_or_HalfHalf = 'half'

    @classmethod
    def makelayout(cls):
        layout = [[sg.Col([
                    [sg.T('フーリエ記述子')],
                    [sg.Frame('Select Description',layout=
                        [[sg.ButtonMenu('Menu',menu_def=F_menu,key='-FDMenu-')],
                         [sg.Text('楕円フーリエ記述子',key='-dp1-')]
                        ])]
                    ],vertical_alignment='top'),Fur.make_double_canvas()]]
        return layout

    def Loop(self,event,values):
        pass

# --------------------------
# --- Class Bezier Curve ---
# --------------------------
# Bezier Curve Fitting
class Bez(RLToolTab):
    WHtoggle = 'half'
    eventlist = {'-WHToggle-':'全周半周切り替え','-FitIt-':'当てはめ実行',
            '-t_sil-':'シルエットオン','-t_curve-':'近似曲線オン',
            '-t_samples-':'標本点オン','-t_cpoint-':'制御点オン'}
    tab = '-tab2-'

    def __init__(self,window):
        super().__init__(window, tab='-tab2-')
        self.setImage(DUMMYSHIL)
        self.WHtoggle = 'half'

    def toggle_HW(self):
        self.WHtoggle = 'half' if self.WHtoggle == 'whole' else 'half'

    @classmethod
    def makelayout(cls):
        layout = [
            [sg.Col([ 
                      [sg.T('ベジエ曲線あてはめ')],
                      [sg.Frame('Parameters',layout=
                        [[sg.Image(filename='./res/picdicpics/Halvesmini.png',key='-WHToggle-', enable_events=True)],
                        [sg.T('Order of curve (N)'),sg.Spin([f"{i:2}" for i in range(3,30)],7,readonly=False,key='-Border-')],
                        [sg.T('Number of Samples (M)')],
                        [sg.Slider(range = (10, 120), default_value=24,orientation = 'h',key='-NSamples-')],
                        [sg.T('Error Tolerance')],
                        [sg.Slider(range = (0.3, 5.0), default_value=2.0,resolution=0.05,orientation = 'h',key='-err_th-')]]
                    )],
                    [sg.Frame('Items on/off',layout=
                        [[sg.Checkbox("Silhouette", key='-t_sil-',default=True),sg.Checkbox("Samples",key='-t_samples-',default=True)],
                        [sg.Checkbox("Curve",key='-t_curve-',default=True),sg.Checkbox("Control Points", key='-t_cpoint-',default=False)]]
                    )],
                    [sg.Image(filename='./res/picdicpics/FitItmini.png',key='-FitIt-', enable_events=True)],
                    [sg.Text('－待機中－',key='-Bez_Message-',text_color='black')]
                ],vertical_alignment='top'),Bez.make_double_canvas()
            ]
        ]
        return layout

    def Loop(self,event,values):
        if event == '-WHToggle-':
            self.WHtoggle = not self.WHtoggle
            self.window.find_element('-WHToggle-').Update(wholeicon if self.WHtoggle else halvesicon)
        elif event == '-FitIt-':
            N = int(values['-Border-'])
            M = int(values['-NSamples-'])
            err_th = values['-err_th-']
            self.window['-Bez_Message-'].update('－計算中－\n表示が変わるまでお待ちください',text_color='orange')
            event, values = self.window.read(timeout=1)
            self.cv_inimgG = cv2.cvtColor(self.cv_inimg,cv2.COLOR_BGR2GRAY)
            self.dp_outimg,funcL,funcR = fitBezier(image=self.cv_inimgG,N=N,Nsamples=M,err_th=err_th)
            drawImage(self.window,img_data=self.dp_outimg,canvas=f'-O_CANVAS2-')
            self.window['-Bez_Message-'].update('\n－待機中－',text_color='black')

# -----------------------------------------------
# --- Class Reshape Straight and Mesurements  ---
# -----------------------------------------------
# 補正形状の計算と計測
class Rfm(RLToolTab):
    eventlist = {'-Straighten-':'補正形状の計算'}
    tab = '-tab3-'
    
    def __init__(self,window):
        super().__init__(window, tab='-tab3-')
        self.setImage(DUMMYSHIL)

    @classmethod
    def makelayout(cls):
        layout = [
            [sg.Col([
                [sg.T('形状補正＆計測')],
                [sg.Frame('Parameters',layout=
                    [[sg.T('Order of curve (N)'),sg.Spin([f"{i:2}" for i in range(4,13)],7,readonly=False,key='-Corder-')],
                    [sg.T('Number of ReSamples (M)')],
                    [sg.Slider(range = (10, 60), default_value=24,orientation = 'h',key='-CSamples-')]]
                )],
                [sg.Image(filename='./res/picdicpics/Straightenmini.png',key='-Straighten-', enable_events=True)],
                [sg.T('Root Length:'),sg.T('',key='-Length-')],
                [sg.T('Max Width:'),sg.T('',key='-Width-')],
                [sg.VerticalSeparator(pad=50)],
                [sg.HorizontalSeparator(pad=1),sg.Text('－待機中－',key='-Rfm_Message-')]
            ],vertical_alignment='top'),
            Rfm.make_double_canvas()]
        ]
        return layout

    # 中心線に垂直となる線分列を求める
    def NormalLadder30(self,img,fl,fr,fc,n_samples=32):
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

    #上下端の接線と輪郭の交点
    def crossPImg0(self,img, fc):
        t = symbols('t')
        fcx,fcy = fc
        dx0, dy0 = float((diff(fcx, t)).subs(t,0)), float((diff(fcy, t)).subs(t,0)) # 上端の傾き 
        dx1, dy1 = float((diff(fcx, t)).subs(t,1)), float((diff(fcy, t)).subs(t,1)) # 下端の傾き
        x0,y0 = int(round(float(fcx.subs(t,0)))),int(round(float(fcy.subs(t,0)))) # 上端
        x1,y1 = int(round(float(fcx.subs(t,1)))),int(round(float(fcy.subs(t,1)))) # 下端
 
        canvas0 = np.zeros_like(img)  # 白紙の描画キャンバス
        con = rd.getContour(img) # 輪郭
        rdimg = canvas0.copy() # 輪郭用
        cv2.drawContours(rdimg, con, -1, 1, thickness=1)

        canvas1 = canvas0.copy()
        x01,y01 = x0 - 20*dx0/dy0 if dy0 != 0 else x0, y0 - 20   # 上端から中心軸に沿って 上に20画素上の点
        x02,y02 = x0 + 20*dx0/dy0 if dy0 != 0 else x0, y0 + 20   # 上端から中心軸に沿って 下に20画素上の点
        x12,y12 = x1 + 20*dx1/dy1 if dy1 != 0 else x1, y1 + 20   # 下端から中心軸に沿って 下に20画素下の点
        x11,y11 = x1 - 20*dx1/dy1 if dy1 != 0 else x1, y1 - 20   # 下端から中心軸に沿って 上に20画素上の点

        x01,y01 = int(round(float(x01))), int(round(float(y01))) # 整数化
        x11,y11 = int(round(float(x11))), int(round(float(y11)))
        x02,y02 = int(round(float(x02))), int(round(float(y02))) # 整数化
        x12,y12 = int(round(float(x12))), int(round(float(y12)))
        canvas1 = cv2.line(canvas1, (x01, y01), (x02, y02),1, 2)  # 幅3（2*2-1）の直線を明るさ１で描く
        canvas1 = cv2.line(canvas1, (x11, y11), (x12, y12),1, 2)  # 幅3（2*2-1）の直線を明るさ１で描く
        canvas = rdimg + canvas1

        # 上の交点　　　重なった場所は値が２となっている.
        cross_pointsU = np.where(canvas[:y02, :] == 2)
        (topx,topy) = int(round(np.mean(cross_pointsU[1]))),int(round(np.mean(cross_pointsU[0])))
        # 下の交点　　　重なった場所は値が２となっている.
        cross_pointsB = np.where(canvas[y11:, :] == 2)
        (bottomx,bottomy) = int(round(np.mean(cross_pointsB[1]))),int(round(np.mean(cross_pointsB[0]))+y11)
        return (topx,topy),(bottomx,bottomy)

    # 補正形状を求める
    def straighten(self,N=6,CSamples=24,topZero=False):
        # global dp_inimg,dp_outimg
        img = self.cv_inimgG
        self.dp_inimg,fl,fr = fitBezier(image=img,N=N,Nsamples=CSamples,err_th=1.0)
        item0 = io.BytesIO()
        rd.drawBez(rdimg=img,stt=0.0,end=1.0,bezL=fl,bezR=fr,saveImage=True,savepath=item0,savesize=(320,320))
        self.dp_inimg = item0.getvalue()
        drawImage(self.window,img_data=self.dp_inimg,canvas=f'-I_CANVAS3-')
        event, values = self.window.read(timeout=1)
        fc0 = (fl+fr)/2 # 仮の中心軸
        dp = rd.getDenseParameters(fc0,n_samples=CSamples,span=0) #  均等間隔になるようなパラメータセットを求める
        samples = [[int(float(fc0[0].subs('t',s))),int(float(fc0[1].subs('t',s)))] for s in dp]
        samples = np.array(samples)
        bezC = BezierCurve(N=4,samples = samples,prefunc=fc0) # ,prefunc=fc は使わなくてもあまり変わらない

        moption = [(samples[0]+samples[1])/2,(samples[-2]+samples[-1])/2]
        bestcpsC, fc1, minerrorC = bezC.fit1T(maxTry=300, mode=1, lr=0.015,lrP=1140,withErr=True, withEC=False,
                    tpara=[], pat=200, err_th=1.0, threstune=1.0,moption = moption)

        rd.drawBez(rdimg=img,stt=0.0,end=1.0,bezL=fl,bezR=fr,bezC=fc1,saveImage=True,savepath=item0,savesize=(320,320))
        self.dp_inimg = item0.getvalue()
        drawImage(self.window,img_data=self.dp_inimg,canvas=f'-I_CANVAS3-')
        event, values = self.window.read(timeout=1)

        lpoints2,rpoints2,cpoints,dpc = self.NormalLadder30(img,fl,fr,fc1,n_samples=CSamples)
        # 左の標本、右の標本、中歯軸の標本、対応するパラメータ

        # 計算不能箇所のデータの補間
        def interporation(plist):
            while np.sum(plist) == np.inf: # np.inf を含むなら除去を繰り返す
                for i in range(len(plist)):
                    if np.sum(plist[i]) == np.inf :
                        print("欠",end="")
                        if (i !=0 and i !=len(plist)-1) and np.sum(plist[i-1]+plist[i+1]) != np.inf: # 当該は無限で、前後は無限ではない場合
                            plist = np.r_[plist[0:i],[[int(round(((plist[i-1]+plist[i+1])/2)[0])),
                                                    int(round(((plist[i-1]+plist[i+1])/2)[1]))]],plist[i+1:]]
                        elif len(plist[i:])>=3 and np.sum(plist[i+1]+plist[i+2]) != np.inf:
                            plist = np.r_[plist[0:i],[plist[i+2]-2*(plist[i+2]-plist[i+1])],plist[i+1:]]
                        elif len(plist[0:i])>=2 and np.sum(plist[i-1]+plist[i-2]) != np.inf:
                            plist = np.r_[plist[0:i],[plist[i-2]-2*(plist[i-2]-plist[i-1])],plist[i+1:]]
            return plist

        # 各サンプル点における幅を求める
        lpoints2 = interporation(lpoints2)
        rpoints2 = interporation(rpoints2)
        width = []
        for [lx,ly],[rx,ry] in zip(lpoints2,rpoints2):
            w = np.sqrt((lx-rx)*(lx-rx)+(ly-ry)*(ly-ry))
            width.append(w)

        # 各サンプル点までの距離を求める
        dp,lengths = rd.getDenseParameters(fc1,n_samples=128,span=0,needlength=True)
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
        if topZero :
            found = [found[0]]+found+[found[-1]]
            topmargin = bottommargin = 0
        else:
            # 図的に中心軸の上端下端を求める
            con = rd.getContour(img) # 輪郭
            rdcimg = np.zeros_like(img)  # 描画キャンバスの準備
            cv2.drawContours(rdcimg, con, -1, 255, thickness=1) # 輪郭線を描く
            (topx,topy),(bottomx,bottomy) = self.crossPImg0(img, fc1) # 上端の接線と輪郭の交点
            t = symbols('t')
            (c0x,c0y),(c1x,c1y) = fc1.subs(t,0),fc1.subs(t,1)
            c0x,c0y = float(c0x),float(c0y)
            c1x,c1y = float(c1x),float(c1y)
            topmargin = np.sqrt((topx-c0x)**2+(topy-c0y)**2) # トップのカットされた長さ
            bottommargin = np.sqrt((bottomx-c1x)**2+(bottomy-c1y)**2) # ボトムのカットされた長さ
            found = [-topmargin]+found+[found[-1]+bottommargin]
            
        # 形状補正データのサンプルの生成
        samples = np.array([[int(round(l)),int(round(w))] for l,w in zip(width,found)])
        bez = BezierCurve(N=N,samples=samples) # インスタンス生成
        moption = [(samples[0]+samples[1])/2,(samples[-2]+samples[-1])/2]
        cps,fc = bez.fit1T(maxTry=1000, mode=1, lr=0.015,lrP=1140,withErr=False, withEC=False,tpara=[], pat=300, err_th=1.0, threstune=1.0,moption=moption)
        # 結果の描画
        gx,gy,(x,y,w,h,a) = rd.getCoG(img)
        # fig0.subplots_adjust(left=0, right=1, bottom=0, top=1) # 余白なし
        item0 = io.BytesIO()
        rd.drawBez(rdimg=img,stt=0.0,end=1.0,bezL=fl,bezR=fr,bezC=fc1,saveImage=True,savepath=item0,savesize=(320,320))
        # plt.cla()
        self.dp_inimg = item0.getvalue()
        item1 = io.BytesIO()
        testsamples = np.array([[s/2,t+y] for s,t in samples])
        rd.drawBez(rdimg=img,stt=0.0,end=1.0,bezL=[-fc[0]/2,y+fc[1]],bezR=[fc[0]/2,y+fc[1]],cntL=testsamples,
                axisoptions=['equal'],saveImage=True,savepath=item1,savesize=(600,600))
        # plt.cla()
        length = topmargin+found[-1]+bottommargin
        width = np.array(width).max()
        print(f"補正後の全長は {length:3.1f} pixels")
        print(f"最大幅は {width:3.1f} pixels")
        self.window['-Length-'].update(f'{length:3.1f}')
        self.window['-Width-'].update(f'{width:3.1f}')   
        self.dp_outimg = item1.getvalue()
        # return fc, found[-1], np.array(width).max()

    def Loop(self,event,values):
        if event == '-Straighten-':
            self.window['-Rfm_Message-'].update(value='－計算中－　表示が変わるまでお待ちください',text_color='orange')
            N = int(values['-Corder-'])
            M = int(values['-CSamples-'])
            event, values = self.window.read(timeout=1)
            self.straighten(N=N,CSamples=M,topZero=False)
            drawImage(self.window,img_data=self.dp_outimg,canvas=f'-O_CANVAS3-')
            drawImage(self.window,img_data=self.dp_inimg,canvas=f'-I_CANVAS3-')
            self.window['-Rfm_Message-'].update(value='－待機中－')

# window の初期化
'''
def initWindow(window):
    setImage(DUMMYPATH,window,tab='-tab0-')
    setImage(DUMMYSHIL,window,tab='-tab1-')
    setImage(DUMMYSHIL,window,tab='-tab2-')
    setImage(DUMMYSHIL,window,tab='-tab3-')
'''

# IMG0,IMG1,IMG2,IMG3 = None,None,None,None
# -----  トップレイアウト定義
menu_def = [['File', ['&Open','&Save', '&Quit']]]

def make_layout():
    layout = [[sg.Menu(menu_def)],
        [sg.TabGroup([[
            sg.Tab('Silhouette Extraction', Seg.makelayout(),key='-tab0-'),
            sg.Tab('Fourier Description', Fur.makelayout(),key='-tab1-'),
            sg.Tab('Bezier Fitting', Bez.makelayout(),key='-tab2-'),
            sg.Tab('Shape Correction & Mesurements', Rfm.makelayout(),key='-tab3-'),
        ]],key='-TAB-')],
        [sg.Text('Image path:'),sg.Text('',key='-path-')]
    ]
    return layout

import time
def message_thread(window):
    while True:
        #　1秒スリープ
        time.sleep(1)
        #　スレッドから、データ（スレッド名、カウンタ）を送信
        window['-Rfm_Message-'].update('TEST')

# Main Widget
if __name__ == '__main__':

    # Window の初期化
    window = sg.Window('Radish Lab Demo',make_layout(), resizable=True, finalize=True,size=(1200,682))
    oldevent,oldvalues = None,None # 1つ前のイベントとその値
    tabSeg = Seg(window) # セグメンテーションのタブ
    tabFur = Fur(window) # フーリエ記述子のタブ
    tabBez = Bez(window) # ベジエ近似のタブ
    tabRfm = Rfm(window) # 形状補正と計測のタブ
    tabs = {'-tab0-':tabSeg,'-tab1-':tabFur,'-tab2-':tabBez,'-tab3-':tabRfm}
    # initWindow(window)

    while True:

        event, values = window.read(timeout=50)
        if oldevent != event:
            print(event)
        oldevent = event
        oldvalues = values

        if event in (None, 'Quit','Exit') or event == sg.WIN_CLOSED:
            break
        # ファイルメニューイベント
        if event == 'Open':
            tab = values['-TAB-']
            path = sg.popup_get_file('Choose Image',initial_folder=SRCTOP,file_types=(("Image File",('*.jpg','*.jpeg','*.png')),)) 
            if path != None:
                tabs[tab].setImage(path)

        # タブ切り替え
        #elif lasttab != values['-TAB-']:
        #    lasttab = values['-TAB-'] 
            # window.close()
            # window = mkwindow(path=path,tab=lasttab) 

        # セグメンテーションタブでの処理
        elif event in Seg.eventlist.keys():
            tabSeg.Loop(event,values)
        # フーリエ記述子タブでの処理
        elif event in Fur.eventlist.keys():
            tabFur.Loop(event,values)
        # ベジエ近似タブでの処理
        elif event in Bez.eventlist.keys():
            tabBez.Loop(event,values)
        # 補正と計測タブでの処理
        elif event in Rfm.eventlist.keys():
            tabRfm.Loop(event,values)
        # フーリエ記述子
        elif event == '-b1-':
            window['-t1-'].update(values['-b1-'])
    
window.close()
