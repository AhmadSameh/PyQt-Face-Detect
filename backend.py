from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from frontend import Ui_MainWindow

import cv2 as cv


class Face_Detector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.capture = False
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.ExtBtn.clicked.connect(self.close_cam)
        self.ui.CapBtn.clicked.connect(self.capture_frame)
    
    def display_image(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888
        gray_face = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        haar_cascade = cv.CascadeClassifier('haar_face.xml')
        faces_rect = haar_cascade.detectMultiScale(gray_face, 1.1, 9, None)
        for (x,y,w,h) in faces_rect:
            cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        if self.capture:
            cv.imwrite('img.jpg', img)
            self.capture = False
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.ui.Cam.setPixmap(QPixmap.fromImage(img))
        self.ui.Cam.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    
    def capture_video(self):
        cap = cv.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv.flip(frame, 1)
            self.display_image(frame, 1)
            cv.waitKey()
        cap.release()
    
    @pyqtSlot()
    def close_cam(self):
        QMainWindow.close()
        cv.destroyAllWindows()
    
    @pyqtSlot()    
    def capture_frame(self):
        self.ui.ConfirmCap.setText("FRAME CAPTURED!")
        self.capture = True
        
        
if __name__ == "__main__":
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    face_detect = Face_Detector()
    face_detect.show()
    face_detect.capture_video()
    sys.exit(app.exec_())
