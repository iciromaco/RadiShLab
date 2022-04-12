# Class Bezier Curbe
# ベジエ曲線のクラス定義
class BezierCurve:
    # インスタンス変数
    # f [X座標関数式,Y座標関数式]
    # samples 標本点のリスト
    # ts 標本点に対するベジエパラメータ

    # クラス変数
    # xx driftThres = 0.01 # 繰り返しにおけるパラメータ変動幅の平均値に対するしきい値
    # xx errorThres = 0.01 # 繰り返しを打ち切る誤差変化量
    dCount = 3  # ２分探索の打ち切り回数 （3以上が望ましい）
    convg_coe = 1e-5 # 5.0e-8  # 収束の見切り　　１回あたりの誤差減少が少なすぎる時の打ち切り条件を決める値 3e-8〜8e-8 が最適
    mloop_itt = 3 # fit1T mode 0 の繰り返しループにおける、minimize() の繰り返し回数。
    debugmode = False
    AsymptoticPriority = 'distance'  # パラメータ更新法
    wandb = False
    openmode = False

    # 'distance':距離優先、'span':間隔優先

    def __init__(self, N=5, samples=[], prefunc=None, tpara=[]):
        self.samples = samples  # 標本点
        self.prefunc = prefunc  # 関数式
        # self.f  ベジエ曲線の式 = [fx,fy]
        # 'px0','px1'... が制御点のX座標を表すシンボル
        # 'py0','py1'...が制御点のY座標を表すシンボル
        self.N = N
        # ベジエ曲線を定義するのに使うシンボルの宣言
        P = [Symbol('P' + str(i)) for i in range(N+1)]  # 制御点を表すシンボル（数値変数ではない）
        px = [var('px'+str(i)) for i in range(N+1)]  # 制御点のx座標を表すシンボル
        py = [var('py'+str(i)) for i in range(N+1)]  # 制御点のy座標を表すシンボル
        t = symbols('t')
        v = var('v')
        for i in range(N+1):     # 制御点のシンボルと成分の対応付け
            P[i] = Matrix([px[i], py[i]])
        # N次のベジエ曲線の定義式制御点 P0～PN とパラメータtの関数として定義
        v = 1-t
        bezf = Matrix([0, 0])
        for i in range(0, N+1):
            bezf = bezf + binomial(N, i)*v**(N-i)*t**i*P[i]
        self.f = bezf
        # もし、inf データが含まれるならば、補間する（計算で求められた座標データがサンプルの場合にありうる）
        if len(samples) > 0:
            self.samples = self.interporation(samples)
        # 初期パラメータのセット
        if len(tpara) > 0:
            self.ts = tpara.copy()
        else:
            self.ts = self.assignPara2Samples(prefunc=prefunc)

    # 当てはめの自乗誤差の平均値を算出する関数
    def f_meanerr(self, fx, fy, ts):
        sps = self.samples
        # fx, fy : t の symfy関数、ts: ベジエのパラメータのリスト, sps サンプル点のリスト
        t = symbols('t')
        nfx, nfy = lambdify(t, fx, "numpy"), lambdify(t, fy, "numpy")
        onps = [[nfx(ts[i]), nfy(ts[i])] for i in range(len(ts))]
        return mean([(sps[i][0]-onps[i][0])**2+(sps[i][1]-onps[i][1])**2 for i in range(len(sps))])

    # 解なしの部分に np.inf が入っているのでその抜けを前後から推定してデータを埋める
    def interporation(self, plist):
        # plist : np.inf が混入している可能性のある座標の numpy array
        foundinf = 0
        while np.sum(plist) == np.inf:  # np.inf を含むなら除去を繰り返す
            for i in range(len(plist)):
                if np.sum(plist[i]) == np.inf:
                    foundinf += 1
                    print("欠", end="")
                    # 当該は無限で、前後は無限ではない場合
                    if (i != 0 and i != len(plist)-1) and np.sum(plist[i-1]+plist[i+1]) != np.inf:
                        plist = np.r_[plist[0:i], [
                            (plist[i-1]+plist[i+1])/2], plist[i+1:]]
                    elif len(plist[i:]) >= 3 and np.sum(plist[i+1]+plist[i+2]) != np.inf:
                        plist = np.r_[plist[0:i], [plist[i+2] -
                                                   2*(plist[i+2]-plist[i+1])], plist[i+1:]]
                    elif len(plist[0:i]) >= 2 and np.sum(plist[i-1]+plist[i-2]) != np.inf:
                        plist = np.r_[plist[0:i], [plist[i-2] -
                                                   2*(plist[i-2]-plist[i-1])], plist[i+1:]]
        if foundinf:
          print("A total of {} data are missing.".format(foundinf))
        return plist

    # サンプル点が等間隔になるようにパラメータを設定する
    def assignPara2Samples(self, prefunc=None, scatter=False):
        samples = self.samples
        if len(samples) == 0:  # 標本点を与えずにベジエ曲線の一般式として使うこともできる
            return
        if prefunc != None:  # 近似式が与えられている場合
            return getDenseParameters(func=prefunc, n_samples=len(samples), span=0)
        # 曲線の式が不明である場合は、単純に０〜１を等間隔に刻んだ値を返す
        else:  # パラメータが与えられていない場合、0～1をリニアに各サンプル点までので経路長で刻む
            # axlength = np.array(cv2.arcLength(samples, False)) # 点列に沿って測った総経路長
            # 各サンプル点の始点からの経路長の全長に対する比を、各点のベジエパラメータの初期化とする
            if scatter:
                axlength = np.array(cv2.arcLength(samples, False))
                return [cv2.arcLength(samples[:i+1],False)  for i in range(len(samples))]/axlength
            else:
                return [i/(len(samples)-1) for i in range(len(samples))]
        

    # 制御点のi番目を代入
    def setACP(self, f, i, cp):
        [x, y] = cp
        sx = var('px'+str(i))
        sy = var('py'+str(i))
        f[0] = f[0].subs(sx, x)
        f[1] = f[1].subs(sy, y)
        return f

    # 制御点座標をセットして関数式を完成
    def setCPs(self, cps):
        f = self.f.copy()
        for i in range(self.N+1):
            self.setACP(f, i, cps[i])
        return f

    # ベジエ近似レベル０（標本点のパラメータを等間隔と仮定してあてはめ）
    def fit0(self, prefunc=None, tpara=[], scatter=False, moption=[], withErr=False):
        # ts 標本点に対するパラメータ割り当て
        samples = self.samples  # 標本点
            
        # t 標本点に結びつけるパラメータは引数として与えられているならそれを、さもなくばリニアに設定
        if len(tpara) > 0: #パラメータが与えられているならそれを使う
            ts = self.ts = tpara.copy()
        elif prefunc != None: # 関数が与えられているなら、それをもとに等間隔になるよう初期パラメータ設定
            ts = self.ts = self.assignPara2Samples(prefunc=prefunc) 
        else: # さもなくば、self.prefunc で等間隔か [0,1] を等間隔　ただしscatter=Trueのときは不等間隔
            ts = self.ts = self.assignPara2Samples(prefunc=None,scatter=scatter) 
        N = self.N  # ベジエの次数
        M = len(samples)  # サンプル数
        # バーンスタイン関数の定義
        if len(moption)>1: # 端点部のみ中間点を追加する
            samples = np.concatenate([[samples[0],moption[0]],samples[1:-1],[moption[-1],samples[-1]]])
            ts = np.concatenate([[ts[0],(ts[0]+ts[1])/2],ts[1:-1],[(ts[-2]+ts[-1])/2,ts[-1]]])
            M = M + 2
        x, y = samples[:, 0], samples[:, 1]  # 標本点のｘ座標列とｙ座標列
            
        def bs(n, t):
            return binomial(N, n)*(1-t)**(N-n)*t**n
        # 標本点と、それに対応する曲線上の点の距離の総和を最小化するような制御点を求める
        if BezierCurve.openmode:
            exA = np.array([[sum([bs(i, ts[k])*bs(n, ts[k]) for k in range(M)])
                             for i in range(N+1)] for n in range(N+1)], 'float64')
            if np.linalg.matrix_rank(exA,tol=1e-20) != exA.shape[0]:
                print("Rank Warning(tol:1e-20)")
            exBX = np.array([[sum([x[k]*bs(n, ts[k]) for k in range(M)])]
                             for n in range(N+1)], 'float64')
            exBY = np.array([[sum([y[k]*bs(n, ts[k]) for k in range(M)])]
                             for n in range(N+1)], 'float64')
            cpsx = np.linalg.solve(exA, exBX)
            cpsy = np.linalg.solve(exA, exBY)

        else:  # 両端点をサンプルの両端に固定する場合
            exA = np.array([[sum([bs(i, ts[k])*bs(n, ts[k]) for k in range(M)])
                             for i in range(1, N)] for n in range(1, N)], 'float64')
            if np.linalg.matrix_rank(exA,tol=1e-20) != exA.shape[0]:
                print("Rank Warning(tol:1e-20)")
            exBX = np.array([[sum([bs(n, ts[k])*(x[k]-x[0]*(1-ts[k])**N - x[-1]*ts[k]**N)
                                   for k in range(M)])] for n in range(1, N)], 'float64')
            exBY = np.array([[sum([bs(n, ts[k])*(y[k]-y[0]*(1-ts[k])**N - y[-1]*ts[k]**N)
                                   for k in range(M)])] for n in range(1, N)], 'float64')
            cpsx = np.r_[[[x[0]]], np.linalg.solve(exA, exBX), [[x[-1]]]]
            cpsy = np.r_[[[y[0]]], np.linalg.solve(exA, exBY), [[y[-1]]]]

        cps = [[i[0][0], i[1][0]] for i in zip(cpsx, cpsy)]
        func = self.setCPs(cps)

        if withErr:
            fx,fy = func
            return cps, func, self.f_meanerr(fx, fy, self.ts)
        else:
            return cps, func

    #  パラメトリック曲線　curvefunc 上で各サンプル点に最寄りの点のパラメータを対応づける
    def refineTparaN(self, bezierparameters, curvefunc, stt, end):
        # bezierparameters 標本点と結びつけられたベジエパラメータ
        # curvefunc 曲線の式
        # stt,end パラメータを割り当てる標本番号の最初と最後（最後は含まない）

        sps = self.samples
        ts = bezierparameters
        f = curvefunc

        def searchband(n):
            if n == 0:
                return 0, ts[1]/2.0
            elif n == len(ts)-1:
                return (ts[-2]+1.0)/2.0, 1
            else:
                return (ts[n-1]+ts[n])/2.0, (ts[n]+ts[n+1])/2.0

        # 曲線 linefunc(t) 上で座標(x,y) に最も近い点のパラメータを2分サーチして探す関数 Numpy化で高速化 2021.04.04
        def nearest(x, y, oldt, curvefunc, pmin, pmax, err_th=0.75, dcount=5):
            # x,y 座標、oldt 暫定割り当てのパラメータ、pmin,pmax 探索範囲、dcount 再起呼び出しの残り回数
            # ベジエ曲線の関数を記号式から numpy 関数式に変換

            def us(p):
                x, y = p
                return np.array([float(x), float(y)])

            def nearestNp(p, oldt, funcX, funcY, pmin, pmax, dcount=5):
                # def nearestNp(p, oldt, funcX, funcY, diffX, diffY, pmin, pmax, dcount=5):
                # sympy 表現の座標を数値化
                epsilon = 1e-07
                ps = funcX(pmin), funcY(pmin)  # パラメータ最小点
                pe = funcX(pmax), funcY(pmax)  # パラメータ最大点
                ls = np.linalg.norm(us(ps) - p)  # pmin と p の距離
                le = np.linalg.norm(us(pe) - p)  # pmax と p の距離
                mid = mid = (le*pmin+ls*pmax)/(ls+le)  # pmin と pmax のパラメータの平均
                pm = funcX(mid), funcY(mid)  # 中間パラメータ点
                lold = funcX(oldt)
                lm = np.linalg.norm(us(pm) - p)  # pmid と p の距離
                m = min([ls, lm, le, lold])
                if m == lold:  # 改善されない
                    return oldt
                elif m == lm:
                    newt = mid
                    dd = min(mid - pmin, pmax - mid)/2.0
                    newpmin = mid - dd
                    newpmax = mid + dd
                elif m == le:
                    newt = pmax
                    newpmin = mid  # (mid + pmax)/2.0
                    newpmax = pmax - epsilon
                else:
                    newt = pmin
                    newpmin = pmin + epsilon
                    newpmax = mid  # (pmin + mid)/2.0
                ddx = funcX(newpmax)-funcX(newpmin)  # x の範囲の見積もり
                ddy = funcY(newpmax)-funcY(newpmin)  # y の範囲の見積もり
                ddv = ddx*ddx + ddy*ddy
                if m < err_th or ddv < err_th*err_th or (dcount <= 0 and m > err_th*2):
                    return newt
                else:
                    return nearestNp(p, newt, funcX, funcY, newpmin, newpmax, dcount-1)

            p = np.array([x, y])
            t = symbols('t')
            (funcX, funcY) = curvefunc  # funcX,funcY は 't' の関数
            # (diffX, diffY) = (diff(funcX, 't'), diff(funcY, 't'))  # 導関数
            # 関数と導関数を numpy 関数化　（高速化目的）
            (funcX, funcY) = (lambdify(t, funcX, "numpy"), lambdify(t, funcY, "numpy"))
            # (diffX, diffY) = (lambdify(t, diffX, "numpy"), lambdify(t, diffY, "numpy"))
            # return nearestNp(p, oldt, funcX, funcY, diffX, diffY, pmin, pmax, dcount=dcount)
            return nearestNp(p, oldt, funcX, funcY, pmin, pmax, dcount=dcount)

        if stt == end:
            return ts

        nmid = (stt+end)//2  # 探索対象の中央のデータを抜き出す
        px, py = sps[nmid]  # 中央のデータの座標
        band = searchband(nmid)
        tmid = ts[nmid]

        midpara = nearest(
            px, py, tmid, f, band[0], band[1], dcount=BezierCurve.dCount)  # 最も近い点を探す

        ts[nmid] = midpara
        ts = self.refineTparaN(ts, f, stt, nmid)
        ts = self.refineTparaN(ts, f, nmid+1, end)

        return ts

    # ベジエ近似　パラメータの繰り返し再調整あり
    def fit1(self, maxTry=0, withErr=False, withEC=False, tpara=[], prefunc=None,pat=300, err_th=0.75, threstune=1.00,scatter=False, moption=[]):
        # maxTry 繰り返し回数指定　0 なら誤差条件による繰り返し停止
        # withErr 誤差情報を返すかどうか  withECも真ならカウントも返す
        # tpara  fit0() にわたす初期パラメータ値
        # prefunc tpara を求める基準となる関数式がある場合は指定
        # pat 300 これで指定する回数最小エラーが更新されなかったら繰り返しを打ち切る
        # threstune 1.0  100回以上繰り返しても収束しないとき、この割合で収束条件を緩める
        # scatter 不等間隔標本の場合に真

        sps = self.samples

        # #######################
        # Itterations start here フィッティングのメインプログラム
        # #######################
        trynum = 0  # 繰り返し回数
        lastgood = -1
        rmcounter = 0  # エラー増加回数のカウンター
        priority = BezierCurve.AsymptoticPriority

        N = self.N
        # 初期の仮パラメータを決めるため、fit0(2N)で近似してみる 
        doubleN = 2*N if N < 9 else 18
        prebez = BezierCurve(N=doubleN,samples=self.samples)
        precps, prefunc = prebez.fit0(prefunc=prefunc, tpara=tpara, scatter=scatter,moption=moption)
        # 仮近似曲線をほぼ等距離に区切るようなパラメータを求める
        # 改めて fit0 でN次近似した関数を初期近似とする
        # cps, func = self.fit0(prefunc = prefunc, tpara=tpara)  # レベル０フィッティングを実行
        cps, func = self.fit0(prefunc = prefunc, scatter=scatter, moption=moption)  # レベル０フィッティングを実行
        #cps, func = self.fit0(tpara=tpara,scatter=scatter,moption=moption)  # レベル０フィッティングを実行
        [fx, fy] = bestfunc = func
        bestcps = cps
        ts = bestts = self.ts.copy()

        minerror = self.f_meanerr(fx, fy, ts=ts)  # 当てはめ誤差
        errq = deque(maxlen=3) # エラーを３回分記録するためのバッファ
        for i in range(2):
          errq.append(np.inf)
        errq.append(minerror)
          
        if BezierCurve.debugmode:
            print("initial error:{:.5f}".format(minerror))

        while True:
            # パラメータの再構成（各標本点に関連付けられたパラメータをその時点の近似曲線について最適化する）
            if priority == 'distance' or priority == 'hyblid':
                ts = self.refineTparaN(ts, [fx, fy], 0, len(sps))
            # 標本点が等間隔であることを重視し、曲線上の対応点も等間隔であるということを評価尺度とする方法
            elif priority == 'span':
                ts = self.assignPara2Samples(prefunc=[fx, fy])

            # レベル０フィッティングを再実行
            cps, func = self.fit0(tpara=ts,scatter=scatter,moption=moption)
            [fx, fy] = func

            # あてはめ誤差を求める
            error = self.f_meanerr(fx, fy, ts=ts)
            old3err = errq.popleft() # ３回前のエラーを取り出し
            errq.append(error) # 最新エラーをバッファに挿入

            if BezierCurve.wandb:
                BezierCurve.wandb.log({"loss": error})
            if error < minerror:
                convg_coe = BezierCurve.convg_coe
                convergenceflag = (trynum - lastgood > 3 or trynum - lastgood == 1) and ((old3err - error)/3.0 < convg_coe*(error-err_th))
                # convergenceflag = True if (minerror - error)/(trynum - lastgood) < convg_coe*(error-err_th) else False
                lastgood = trynum
                bestts = ts.copy()  # 今までで一番よかったパラメータセットを更新
                bestfunc = func  # 今までで一番よかった関数式を更新
                minerror = error  # 最小誤差を更新
                bestcps = cps  # 最適制御点リストを更新
                print(".", end='')
            else:
                print("^", end='')

            # 繰り返し判定調整量
            # 繰り返しが100回を超えたら条件を緩めていく
            thresrate = 1.0 if trynum <= 100 else threstune**(trynum-100)
            if BezierCurve.debugmode:
                print("{} err:{:.5f}({:.5f}) rmcounter {})".format(
                    trynum, error, minerror, rmcounter))

            rmcounter = 0 if error <= minerror else rmcounter + 1  # エラー増加回数のカウントアップ　減り続けているなら０
            # if convergenceflag or error < err_th*thresrate or rmcounter > pat:
            if (convergenceflag and (trynum > 50)) or error < err_th*thresrate or ((trynum > 100) and rmcounter > pat):
                # pat回続けてエラーが増加したらあきらめる デフォルトは10
                #if (convergenceflag and (trynum > 50)):
                if convergenceflag:
                  print('C')
                elif error < err_th*thresrate :
                  print('E')
                else:
                  print('P')
                if BezierCurve.debugmode:
                    if rmcounter > pat:
                        print("W")
                    else:
                        print("M")
                if priority == 'hyblid':
                    rmcounter = 0
                    priority = 'span'
                else:
                    break

            trynum += 1
            if trynum % 100 == 0:
                print("")
            if maxTry > 0 and trynum >= maxTry:
                break

        self.ts = bestts.copy()
        
        print("")
        if withErr:
            if withEC:
                return bestcps, bestfunc, (minerror,trynum)
            else:
                return bestcps, bestfunc, minerror
        else:
            return bestcps, bestfunc

    # fit1 の tensorflowによる実装
    # def fit1T(self, mode=1, maxTry=0, withErr=False, withEC=False,prefunc=None,tpara=[], lr=0,  lrP=0, pat=300, err_th=1.0, threstune=1.00, trial=None, scatter=False, moption=[]):
    def fit1T(self, mode=1, maxTry=0, withErr=False, withEC=False,prefunc=None,tpara=[], lr=0,  lrP=0, pat=300, err_th=1.0, threstune=1.00, scatter=False, moption=[]):
        # maxTry 繰り返し回数指定　0 なら誤差条件による繰り返し停止
        # withErr 誤差情報を返すかどうか withEC が真ならカウントも返す
        # tpara  fit0() にわたす初期パラメータ値
        # mode 0: 制御点とパラメータを両方同時に tensorflow で最適化する
        # mode 1: パラメータの最適化は tensorflow で、制御点はパラメータを固定して未定係数法で解く
        # fit1T では priority=distance のみ考え、span は考慮しない
        # lr オプティマイザーの学習係数（媒介変数 ts 用）
        # lrP 制御点用オプティマイザーの学習係数の倍率 lr*lrP を制御点の学習係数とする。
        # prefunc tpara を求める基準となる関数式がある場合は指定
        # pat 10 これで指定する回数最小エラーが更新されなかったら繰り返しを打ち切る
        # err_th 0.75  エラーの収束条件
        # threstune 1.0  100回以上繰り返しても収束しないとき、この割合で収束条件を緩める
        # trial Optuna のインスタンス
        # サンプル間隔が等間隔でない場合、scatter=True
        
        sps = self.samples.copy()
        
        # #######################
        # Itterations start here フィッティングのメインプログラム
        # #######################
        trynum = 0  # 繰り返し回数
        lastgood = -1
        rmcounter = 0  # エラー増加回数のカウンター
        priority = BezierCurve.AsymptoticPriority

        N = self.N
        # 初期の仮パラメータを決めるため、fit0(2N)で近似してみる 
        doubleN = 2*N  if N < 9 else 18
        prebez = BezierCurve(N=doubleN,samples=self.samples)
        precps, prefunc = prebez.fit0(prefunc=prefunc, tpara=tpara,scatter=scatter, moption=moption)
        # 仮近似曲線をほぼ等距離に区切るようなパラメータを求める
        # 改めて fit0 でN次近似した関数を初期近似とする
        # cps, func = self.fit0(prefunc = prefunc, tpara=tpara)  # レベル０フィッティングを実行
        cps, func = self.fit0(prefunc = prefunc, scatter=scatter, moption=moption)  # レベル０フィッティングを実行
        (fx,fy) = bestfunc = func
        bestcps = cps
        ts = bestts = self.ts.copy()
        m_ts = self.ts.copy()
        if len(moption)>1: # 端点部のみ中間点を追加する
            sps = np.concatenate([[sps[0],moption[0]],sps[1:-1],[moption[-1],sps[-1]]])
            m_ts = np.concatenate([[m_ts[0],(m_ts[0]+m_ts[1])/2],m_ts[1:-1],[(m_ts[-2]+m_ts[-1])/2,m_ts[-1]]])
        x_data = sps[:,0]
        y_data = sps[:,1]

        minerror = self.f_meanerr(fx, fy, ts=ts)  # 初期当てはめ誤差の算出
        errq = deque(maxlen=3) # エラーを３回分記録するためのバッファ
        for i in range(2):
          errq.append(np.inf)
        errq.append(minerror)

        if BezierCurve.debugmode:
            print("initial error:{:.5f}".format(minerror))

        # tensorflow の変数
        if mode == 0:
            Px = tf.Variable([cps[i+1][0]
                              for i in range(N-1)], dtype='float32')
            Py = tf.Variable([cps[i+1][1]
                              for i in range(N-1)], dtype='float32')

        tts = tf.Variable(m_ts, dtype='float32') # 端点以外のベジエパラメータをtensorflow変数化
        
        # Optimizer 
        # 実験結果としてAdamが優れていたのでAdamを採用。Adam 以外の、経験的な適正パラメータは以下の通り
        # 'AMSgrad':[0.005,650],'Adagrad':[0.02,250],
        # 'Adadelta':[0.013,3500],'Nadam':[0.001,500], 'Adamax':[0.0075,1000],
        # 'RMSprop':[0.0003,1300],'SGD':[5e-6,2e6],'Ftrl':[0.12,1000]}
        default_lrs={'Adam':[0.0015,1140]}        
        if lr == 0: lr = default_lrs['Adam'][0]
        if lrP == 0: lrP = default_lrs['Adam'][1]
        opt = tf.optimizers.Adam(learning_rate=lr) 
        optP = tf.optimizers.Adam(learning_rate=lr*lrP) 

        tfZERO = tf.constant(0.0,tf.float32)
        tfONE = tf.constant(1.0,dtype=tf.float32)
        
        while True:
            olderror = self.f_meanerr(fx, fy, ts=ts) 
            for loopc in range(BezierCurve.mloop_itt):
                # パラメータの再構成（各標本点に関連付けられたパラメータをその時点の近似曲線について最適化する）
                # 関数化したかったが、tape をつかっているせいなのか、エラーがでてできなかった
                with tf.GradientTape(persistent=True) as metatape:
                    vs = tfONE-tts
                    # tts**0 と (1-tts)**0 を含めると微係数が nan となるのでループから外している
                    bezfx = vs**N*cps[0][0]
                    bezfy = vs**N*cps[0][1] 
                    for i in range(1, N):
                        if mode == 0:
                            bezfx = bezfx + comb(N, i)*vs**(N-i)*tts**i*Px[i-1]
                            bezfy = bezfy + comb(N, i)*vs**(N-i)*tts**i*Py[i-1]                                
                        elif mode == 1:
                            bezfx = bezfx + comb(N, i)*vs**(N-i)*tts**i*cps[i][0]
                            bezfy = bezfy + comb(N, i)*vs**(N-i)*tts**i*cps[i][1]
                    bezfx = bezfx + tts**N*cps[N][0]
                    bezfy = bezfy + tts**N*cps[N][1]
                    
                    sqx,sqy = tf.square(bezfx - x_data),tf.square(bezfy - y_data)

                    meanerrx = tf.reduce_mean(tf.square(bezfx - x_data))
                    meanerry = tf.reduce_mean(tf.square(bezfy - y_data))
                    gloss = tf.add(meanerrx, meanerry)                                      

                # ts を誤差逆伝搬で更新
                opt.minimize(gloss, tape=metatape, var_list=tts)
                # ts を更新
                m_ts = tts.numpy() 
                m_ts[0] = 0.0 # 両端は0,1にリセット
                m_ts[-1] = 1.0 
                # check order and reorder 順序関係がおかしい場合強制的に変更       
                ec = 0
                for i in range(1,len(m_ts)-1):
                    if m_ts[i] <= m_ts[i-1]:
                        ec += 1
                        m_ts[i] = m_ts[i-1] + 1e-6
                    if m_ts[i] >= 1.0-(len(m_ts)-i-2)*(1e-6):
                        ec += 1
                        m_ts[i] = min(1.0-(len(m_ts)-i-2)*(1e-6), m_ts[i+1]) - 1e-6
                if ec > 0:
                    print("e%d" % (ec),end="")
            # tts を更新
            tts.assign(m_ts)
            if len(moption)>0:
                ts[1:-1] = m_ts[2:-2].copy()
            else:
                ts = m_ts.copy()
            self.ts = ts.copy()

            if mode == 0:
                # 上で求めたベジエパラメータに対し制御点を最適化
                for loopc in range(BezierCurve.mloop_itt):
                    with tf.GradientTape(persistent=True) as metatape:
                        vs = tfONE-tts
                        # tts**0 と (1-tts)**0 を含めると微係数が nan となるのでループから外している
                        bezfx = vs**N*cps[0][0]
                        bezfy = vs**N*cps[0][1]
                        for i in range(1, N):
                            bezfx = bezfx + comb(N, i)*vs**(N-i)*tts**i*Px[i-1]
                            bezfy = bezfy + comb(N, i)*vs**(N-i)*tts**i*Py[i-1]
                        bezfx = bezfx + tts**N*cps[N][0]
                        bezfy = bezfy + tts**N*cps[N][1]

                        meanerrx = tf.reduce_mean(tf.square(bezfx - x_data))
                        meanerry = tf.reduce_mean(tf.square(bezfy - y_data))
                        gloss = tf.add(meanerrx, meanerry)                

                    optP.minimize(gloss, tape=metatape, var_list=[Px, Py])

                for i in range(1, N):
                    cps[i][0] = Px[i-1].numpy()
                    cps[i][1] = Py[i-1].numpy() 
                func = self.setCPs(cps)
            elif mode == 1:
                cps, func = self.fit0(tpara=ts,scatter=scatter,moption=moption)

            # 誤差評価
            fx,fy = func
            error = self.f_meanerr(fx, fy, ts=ts) 
            old3err = errq.popleft() # ３回前のエラーを取り出し
            errq.append(error) # 最新エラーをバッファに挿入
            convg_coe = BezierCurve.convg_coe # if mode == 1 else BezierCurve.convg_coe/10.0 # 収束判定基準
            convergenceflag = (trynum - lastgood > 3 or trynum - lastgood == 1) and ((old3err - error)/3.0 < convg_coe*(error-err_th))
            
            if BezierCurve.wandb:
                BezierCurve.wandb.log({"loss":error})
            if error < minerror: 
                if trynum - lastgood > 3:
                  convergenceflag = False
                bestts = ts.copy()  # 今までで一番よかったパラメータセットを更新
                bestfunc = func  # 今までで一番よかった関数式を更新
                minerror = error  # 最小誤差を更新
                bestcps = cps  # 最適制御点リストを更新
                print(".", end='')                  
                lastgood = trynum
                rmcounter = 0
            else:
                rmcounter = rmcounter + 1 # エラー増加回数のカウントアップ　減り続けているなら０
                convergenceflag = False
                print("^", end='') 

            # 繰り返しが100回を超えたら条件を緩めていく
            thresrate = 1.0 if trynum <= 100 else threstune**(trynum-100)
            if BezierCurve.debugmode:
                print("{} err:{:.5f}({:.5f}) rmcounter {})".format(
                    trynum, error, minerror, rmcounter))
            
            if (convergenceflag and (trynum > 50)) or error < err_th*thresrate or ((trynum > 100) and rmcounter > pat):
                # pat回続けてエラーが増加したらあきらめる デフォルトは１00 （fit1T は繰り返し1回あたりの変動が小さい）
                if (convergenceflag and (trynum > 50)):
                  print('C')
                elif error < err_th*thresrate :
                  print('E')
                else:
                  print('P')

                if BezierCurve.debugmode:
                    if rmcounter > pat:
                        print("W")
                    else:
                        print("M")
                break
            
            trynum += 1
            if trynum % 100 == 0:
                print("")
            if maxTry > 0 and trynum >= maxTry:
                break
            # Optuna を使っている場合の打ち切り
            '''
            if trial:
                trial.report(error,trynum)
                if trial.should_prune() or error > 1e3:
                    print("Optuna による打ち切り")
                    raise optuna.TrialPruned()
            '''    
        self.ts = bestts.copy()
        fx,fy=bestfunc
            
        print("")
        if withErr:
            if withEC:
                return bestcps, bestfunc, (minerror,trynum)
            else:
                return bestcps, bestfunc, minerror
        else:
            return bestcps, bestfunc

    def fit2(self, mode=0, cont = [], Nprolog=3, Nfrom=5, Nto=12, preTry=200, maxTry=0, lr=0.001, lrP=30000, pat=300, err_th=0.75, threstune=1.0, withErr=False, withEC=False, tpara=[], withFig=False, moption=[]):
        # mode 0 -> fit1() を使う, mode 1 -> fit1T(mode=1)を使う, mode 2 -> fit1T(mode=0) を使う
        # cont 与えられている場合オーバフィッティング判定を行う
        # Nplolog 近似準備開始次数　この次数からNfrom-1までは maxTry 回数で打ち切る
        # Nfrom 近似開始次数　この次数以降は収束したら終了
        # Nto 最大近似次数 Nto < Nfrom  の場合は誤差しきい値による打ち切り
        # maxTry 各次数での繰り返し回数
        # prefunc 初期近似関数
        # err_th 打ち切り誤差
        # pat この回数エラーが減らない場合はあきらめる
        # withErr 誤差と次数を返すかどうか

        Nprolog = Nfrom if Nprolog == 0 else Nfrom
        Ncurrent = Nprolog - 1
        func = self.prefunc
        ts = tpara
        err = err_th + 1
        results = {}
        odds = []
        while Ncurrent < Nto and (err_th < err or len(odds) > 0):
            Ncurrent = Ncurrent + 1
            # abez = BezierCurve(N=Ncurrent, samples=self.samples, tpara=ts, prefunc=func)
            abez = BezierCurve(N=Ncurrent, samples=self.samples, tpara=[], prefunc=None)
            print(Ncurrent, end="")
            # 最大 maxTry 回あてはめを繰り返す
            if mode == 0:
                cps, func, err = abez.fit1(
                    maxTry=preTry if Ncurrent < Nfrom else maxTry, withErr=True, tpara=[], pat=pat, err_th=err_th, threstune=threstune, moption=moption)
            elif mode == 1:
                cps, func, err = abez.fit1T(
                    mode=1, maxTry=preTry if Ncurrent < Nfrom else maxTry, lr=lr, lrP=lrP,withErr=True, tpara=[], pat=pat, err_th=err_th, threstune=threstune, moption=moption)
            elif mode == 2:
                cps, func, err = abez.fit1T(
                    mode=0, maxTry=preTry if Ncurrent < Nfrom else maxTry, lr=lr, lrP=lrP,withErr=True, tpara=[], pat=pat, err_th=err_th, threstune=threstune, moption=moption)
            ts = abez.ts
            if len(cont)>0:
                odds = isOverFitting(func=func,ts=ts,cont=cont,err_th=err_th,of_th=1.0) 
            if err_th >= err and len(odds) > 0:
                print("Order ",Ncurrent," is Overfitting",odds)
            results[str(Ncurrent)] = (cps, func, err,ts)
            # 次数を上げてインスタンス生成
        self.ts = ts.copy()
        print(err,end="")
        if withErr:
            return Ncurrent, results
        else:
            return cps, func

    # デバッグモードのオンオフ
    def toggledebugmode(set=True, debug=False):
        if set:
            BezierCurve.debugmode = debug
        else:  # set が False のときはトグル反応
            BezierCurve.debugmode = not BezierCurve.debugmode
        print("debugmode:", BezierCurve.debugmode)

    # パラメータのセットと表示　引数なしで呼ぶ出せば初期化
    def setParameters(priority='distance', dCount=3, convg_coe=1e-5,swing_penalty=0.0,smoothness_coe=0.0, debugmode=False, openmode=False,wandb=None):

        BezierCurve.AsymptoticPriority = priority  # パラメータ割り当てフェーズにおける評価尺度

        # xx BezierCurve.driftThres = driftThres # 繰り返しにおけるパラメータ変動幅の平均値に対するしきい値
        # xx BezierCurve.errorThres = errorThres # 繰り返しにおける誤差変動幅に対するしきい値
        BezierCurve.dCount = dCount  # サンプル点の最寄り点の2分探索の回数
        BezierCurve.debugmode = debugmode
        BezierCurve.openmode = openmode
        BezierCurve.wandb = wandb
        BezierCurve.convg_coe = convg_coe
        BezierCurve.swing_penalty = swing_penalty
        BezierCurve.smoothness_coe = smoothness_coe
        print("AsymptoticPriority : ", priority)
        print("dCount    : ", dCount)
        print("debugmode : ", debugmode)
        print("openmode  : ", openmode)
        print("wandb  : ", wandb)
        print("convg_coe :", convg_coe)
        print("swing_penalty :", swing_penalty)
        print("smoothness_coe :", smoothness_coe)
        print("")
