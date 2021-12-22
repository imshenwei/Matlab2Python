import sys
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import *  # noqa
from PyQt5.QtWidgets import QMessageBox, QDesktopWidget
from PyQt5.QtGui import QPainter, QImage, QPixmap   # noqa
from resources.demo_GUI import Ui_MainWindow  # noqa
import cv2
import numpy as np
from numpy import *  # noqa
import time
from Beamforming_DAS_demo import *
import resources.record_and_play.mic_array_api as mic


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()

        # UI
        self.setupUi(self)
        self.move_center()
        #
        self.open_camera()
        self.beamforming_init()
        # self.mics = mic.mic_array()

        # B1槽连接 pushButton
        self.pushButton.setText('Algorithm Start')
        self.pushButton.clicked.connect(self.camera_switch)

        # B2槽连接
        self.B2.setText('Close Window')
        self.B2.clicked.connect(self.closeWindow)

        # Painter
        self.painter = QPainter(self)

    def paintEvent(self, a0: QtGui.QPaintEvent):
        if self.open_flag:
            time_paintallstart = time.time()
            _, frame = self.cap.read()  # * 读取摄像头当前帧frame和布尔值_
            # 查看cv2.resize:https://blog.csdn.net/qwert15135156501/article/details/104534131
            # *基于局部像素的重采样g
            # todo 后期根据调试改变长和宽
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR2RGB
            frame = self.testDAS(frame)
            # 创建QImage的对象
            # setpixel, 调色盘setcolor https://blog.csdn.net/seniorwizard/article/details/111309598
            self.Qframe = QImage(
                frame.data, frame.shape[1], frame.shape[0], frame.shape[1]*3, QImage.Format_RGB888)

            self.label_2.setPixmap(QPixmap.fromImage(self.Qframe))  # 投影到label2
            time_paintallend = time.time()
            print('paintall cost', time_paintallend-time_paintallstart)
            print('一秒', 1/(time_paintallend-time_paintallstart), '帧')
            self.update()

    def showImage(self):
        flag, self.cap_im = self.cap.read()
        image_height, image_width, image_depth = self.cap_im.shape
        QIm = cv2.cvtColor(self.cap_im, cv2.COLOR_BGR2RGB)
        QIm = QImage(QIm, image_width, image_height,       # 创建QImage格式的图像，并读入图像信息
                     image_width * image_depth,
                     QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(QIm))

    def move_center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeWindow(self):
        self.close()

    def open_camera(self):
        self.CAM_NUM = 0
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.open_flag = self.cap.open(self.CAM_NUM)  # 摄像头: True打开, False关闭
        if self.open_flag is False:
            print('Please check camera!')

    def camera_switch(self):
        if self.open_flag:  # 关闭
            self.DSP_A()
            self.cap.release()
            self.label_2.clear()
            self.pushButton.setText('Open Camera')
            self.init_imag()
        else:
            flag = self.cap.open(self.CAM_NUM)
            if flag is False:
                mas = QMessageBox.Warning(self, u'Warning', u'Plz check camera!',  # noqa
                                          buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
            else:
                self.pushButton.setText('Algorithm Start')

        self.open_flag = bool(1-self.open_flag)

    def init_imag(self):
        # Im = cv2.imread('dog.jpg')
        Im = cv2.imread('XMU2.jpg')

        # opencv读图片是BGR，qt显示要RGB，
        QIm = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)

        image_height, image_width, image_depth = Im.shape     # 获取图像的高，宽以及深度。

        QIm = QImage(QIm.data, image_width, image_height,       # 创建QImage格式的图像，并读入图像信息
                     image_width * image_depth,
                     QImage.Format_RGB888)

        self.label_2.setPixmap(QPixmap.fromImage(QIm))  # 显示图像

    def DSP_A(self):

        SPL = np.load('testSPL.npy')  # SPL data
        maxSPL = ceil(np.max(SPL))
        minSPL = floor(np.min(SPL))
        print([maxSPL, minSPL])

        count = 0
        time1 = 0
        time2 = 0
        time3 = 0

        self.ret, self.frame = self.cap.read()
        image_height, image_width, image_depth = self.frame.shape
        pic3 = np.array(np.zeros((image_height, image_width, 3)))
        pic1 = np.array(np.zeros((81, 81)))
        color1 = [0, 255, 200, 100, 50]
        start = time.time_ns()

        # self.frame = self.frame[:, 80:560, :]
        end1 = time.time_ns()

        pic1[np.where(SPL > maxSPL-1)] = color1[1]
        pic1[np.where(SPL > maxSPL-2)
             and np.where(SPL <= maxSPL-1)] = color1[2]
        pic1[np.where(SPL > maxSPL-3)
             and np.where(SPL <= maxSPL-2)] = color1[3]
        pic1[np.where(SPL > maxSPL-4)
             and np.where(SPL <= maxSPL-3)] = color1[4]
        pic1[np.where(SPL <= maxSPL-4)] = color1[0]

        # pic2=np.kron(pic1[0:80,0:80],np.ones((6,6)))
        pic2 = cv2.resize(pic1, (image_width, image_height),
                          interpolation=cv2.INTER_AREA)

        end2 = time.time_ns()
        pic3[:, :, 0] = pic2  # RGB通道

        self.hit_img = cv2.add(uint8(pic3), uint8(self.frame))
        end3 = time.time_ns()
        cv2.imwrite("XMU2.jpg", self.hit_img)
        # cv2.imshow("frame", self.hit_img)

    def testDAS(self, frame):
        self.frame = frame

        SPL = self.beamforming()
        maxSPL = ceil(np.max(SPL))
        minSPL = floor(np.min(SPL))
        print([maxSPL, minSPL])

        count = 0
        time1 = 0
        time2 = 0
        time3 = 0
        image_height, image_width, image_depth = self.frame.shape
        pic3 = np.array(np.zeros((image_height, image_width, 3)))
        pic1_width = max(SPL.shape)
        # pic_width和分辨率大小 反比成型区域大小; 区域大小正比于扫描平面大小
        pic1 = np.array(np.zeros((pic1_width, pic1_width)))
        color1 = [0, 255, 200, 100, 50]
        start = time.time_ns()

        # self.frame = self.frame[:, 80:560, :]
        end1 = time.time_ns()

        pic1[np.where(SPL > maxSPL-1)] = color1[1]
        pic1[np.where(SPL > maxSPL-2)
             and np.where(SPL <= maxSPL-1)] = color1[2]
        pic1[np.where(SPL > maxSPL-3)
             and np.where(SPL <= maxSPL-2)] = color1[3]
        pic1[np.where(SPL > maxSPL-4)
             and np.where(SPL <= maxSPL-3)] = color1[4]
        pic1[np.where(SPL <= maxSPL-4)] = color1[0]

        # pic2=np.kron(pic1[0:80,0:80],np.ones((6,6)))
        pic2 = cv2.resize(pic1, (image_width, image_height),
                          interpolation=cv2.INTER_AREA)

        end2 = time.time_ns()
        pic3[:, :, 0] = pic2  # RGB通道

        self.hit_img = cv2.add(uint8(pic3), uint8(self.frame))
        return self.hit_img

    def beamforming_init(self):
        self.z_source = 1  # 麦克风阵列平面与扫描屏幕的距离

        # 麦克风阵列限定区域
        self.mic_r = 0.5  # 麦克风阵列限定区域半径
        self.mic_x = np.array([-self.mic_r, self.mic_r])
        self.mic_y = np.array([-self.mic_r, self.mic_r])
        # 扫描声源限定区域
        self.scan_r = 1  # self.z_source / 2  # 扫描声源限定区域半径
        self.scan_x = np.array([-self.scan_r, self.scan_r])
        self.scan_y = np.array([-self.scan_r, self.scan_r])
        self.c = 343  # 声速
        self.scan_resolution = 0.05  # 扫描网格的分辨率
        # 确定扫描频段（800-4000 Hz）
        self.search_freql = 800
        self.search_frequ = 4000
        # 设定信号持续时间
        self.t_start = 0
        self.t_end = 0.02  # 0.011609977324263039
        self.framerate = 48000  # 44100
        # 导入麦克风阵列
        path_full = '修改代码/resources/6_spiral_array.mat'  # 须要读取的mat文件路径
        # path_full = '修改代码/resources/56_spiral_array.mat'
        self.mic_pos, self.mic_centre, self.mic_x_axis, self.mic_y_axis = get_micArray(
            path_full)
        time_steerVector_start = time.time()

        # *steerVector算法
        self.freqs, self.N_freqs, self.freq_sels = freqs_precaulate(
            self.search_freql, self.search_frequ, self.framerate, self.t_end)
        self.g = steerVector(self.z_source, self.freqs, [self.scan_x, self.scan_y],
                             self.scan_resolution, self.mic_pos.T, self.c, self.mic_centre)
        time_steerVector_end = time.time()
        print('steerVector cost', time_steerVector_end-time_steerVector_start)

    def beamforming(self):
        """DAS 波束成像算法（扫频模式）Delay Summation Algorithm"""

        #  信号的采样频率
        # 引用: https://www.cnblogs.com/xingshansi/p/6799994.html

        # save_wav()
        # wav_path = "修改代码/resources/output_test8.wav"
        # self.framerate, nframes, mic_signal = get_micSignal_from_wav(wav_path)

        # draw_mic_array(mic_x_axis, mic_y_axis)

        mic_signal = simulateMicsignal(
            self.mic_pos, self.z_source, self.c, self.framerate, self.mic_centre, self.t_start, self.t_end)

        # mic_signal = self.mics.get_data().T
        time_start_total = time.time()
        time_start = time.time()
        CSM = developCSM(mic_signal.T, self.search_freql,
                         self.search_frequ, self.framerate, self.t_start, self.t_end, self.N_freqs, self.freq_sels)  # micsignal.shape==6,512
        time_end = time.time()
        print('csm cost', time_end-time_start)

        # 波束成像 -- DAS算法
        time_start = time.time()
        [X, Y, B] = DAS(CSM, self.g, self.freqs, [
                        self.scan_x, self.scan_y], self.scan_resolution)
        time_end = time.time()
        print('DAS cost', time_end-time_start)

        # % 声压级单位转换
        B[B < 0] = 0
        eps = np.finfo(np.float64).eps
        SPL = 20*np.log10((eps+np.sqrt(B.real))/2e-5)
        time_end_total = time.time()
        plot_figure(X, Y, SPL)
        print('totally cost', time_end_total-time_start_total)
        return SPL


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # MainWindow = QMainWindow()
    window = mywindow()
    window.show()

    sys.exit(app.exec_())
