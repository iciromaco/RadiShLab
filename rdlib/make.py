# make.py
import PyInstaller.__main__
PyInstaller.__main__.run([
    'RadishLabTools.py','--onefile','--nowindow','--clean',
    '--add-binary="./_audio_microfrontend_op.so;tensorflow/lite/experimental/microfrontend/python/ops/"',
    '--add-data="./RDSamples/*;RDSamples"',
    '--add-data="./res/picdicpics/*;res/picdicpics"',
    '--add-data="./res/*;res',
    '--hidden-import="PIL.Image"',
    '--hidden-import="PIL.ImgeTk"',
    '--hidden-import="cv2"',
    '--hidden-import="matplotlib.pyplot"',
    '--hidden-import="tensorflow"',
    '--hidden-import="keras.api"',
    '--hidden-import="keras.api._v2"',
    '--hidden-import="rdlib"',
    '--icon="./radishBW32.ico"'
])