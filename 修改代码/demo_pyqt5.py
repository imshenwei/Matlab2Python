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
from Beamforming_DAS_demo import beamforming as DAS_Algorithm


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()

        # UI
        self.setupUi(self)
        self.move_center()

        # open camera
        self.open_camera()

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

        SPL = DAS_Algorithm()
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # MainWindow = QMainWindow()
    window = mywindow()
    window.show()

    # 判断
    sys.exit(app.exec_())
