# !/user/bin/env python3
# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from main_window import *
from child_window import *
from predict import *
import cv2
import os


class ChildWindow(QMainWindow, Ui_ChildWindow):
    def __init__(self, parent=None):
        super(ChildWindow, self).__init__(parent)
        self.ui = Ui_ChildWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.open_folder)  # 为按钮绑定点击事件
        self.ui.comboBox.currentIndexChanged.connect(self.algorithm_changed)
        self.ui.resetButton.clicked.connect(self.reset)
        self.ui.resetButton_2.clicked.connect(self.close_window)
        self.ui.segPic.clicked.connect(self.on_algorithm_changed)

    def close_window(self):
        self.close()

    def open_folder(self):
        options = QFileDialog.Options()
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹", options=options)
        if folder_path:
            self.ui.textBrowser.setText(folder_path)
        self.root = folder_path
        self.imgPath = list(map(lambda x: os.path.join(self.root, x), os.listdir(self.root)))

    def reset(self): # 重置所有相关的状态和控件
        self.ui.lineEdit.clear()
        self.ui.comboBox.setCurrentIndex(0)  # 重置算法选择
        self.ui.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                    "border: 1px solid rgb(224, 130, 43);")
        self.ui.lineEdit.setEnabled(True)
        self.ui.textBrowser.clear()   # 清空 QTextBrowser 中的内容
        self.ui.textBrowser_2.clear()  # 清空 QTextBrowser 中的内容

    def algorithm_changed(self): # 根据选择的算法切换输入框的可编辑状态
        selected_algorithm = self.ui.comboBox.currentText()
        if selected_algorithm == '阈值分割':  # 根据选择的算法执行相应的操作
            self.ui.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                        "border: 1px solid rgb(224, 130, 43);")
            self.ui.lineEdit.setEnabled(True)
            pass
        else:
            self.ui.lineEdit.setStyleSheet("background-color: rgb(239, 239, 239);\n"
                                        "border: 1px solid rgb(224, 130, 43);")
            self.ui.lineEdit.setEnabled(False)
            self.ui.lineEdit.clear()
            pass

    def on_algorithm_changed(self): # 处理算法选择变化的事件
        selected_algorithm = self.ui.comboBox.currentText()
        if selected_algorithm == '阈值分割':  # 根据选择的算法执行相应的操作
            outText1 = "执行阈值分割 阈值设置为" + self.ui.lineEdit.text() + "\n"  # 输出文本到 QTextBrowser
            self.ui.textBrowser_2.insertPlainText(outText1)
            for index in range(len(self.imgPath)):
                img = io.imread(self.imgPath[index])  # numpy.ndarray
                self.img8bit = transfer_16bit_to_8bit(img)
                threshold_value = int(self.ui.lineEdit.text())
                _, binary_image = cv2.threshold(self.img8bit, threshold_value, 255, cv2.THRESH_BINARY)  # 对图像进行二值化
                dirname = self.root + '-predict'
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                filename = dirname + '\\' + self.imgPath[index].split('\\')[-1]
                # cv2.imwrite(filename, binary_image)
                cv2.imencode(filename[-4:], binary_image)[1].tofile(filename)
                outText1_ = "保存图像：" + filename + "\n"  # 输出文本到 QTextBrowser
                self.ui.textBrowser_2.insertPlainText(outText1_)
            pass

        elif selected_algorithm == 'OTSU分割':  # 根据选择的算法执行相应的操作
            outText1 = "执行OTSU分割\n" # 输出文本到 QTextBrowser
            self.ui.textBrowser_2.insertPlainText(outText1)
            for index in range(len(self.imgPath)):
                img = io.imread(self.imgPath[index])  # numpy.ndarray
                self.img8bit = transfer_16bit_to_8bit(img)
                _, binary_image = cv2.threshold(self.img8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 对图像进行二值化
                dirname = self.root + '-predict'
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                filename = os.path.join(dirname, self.imgPath[index].split('\\')[-1])
                # cv2.imwrite(filename, binary_image)
                cv2.imencode(filename[-4:], binary_image)[1].tofile(filename)
                outText1_ = "保存图像：" + filename + "\n"
                self.ui.textBrowser_2.insertPlainText(outText1_)
            pass

        elif selected_algorithm == '亮点分割':  # 根据选择的算法执行相应的操作
            outText1 = "执行亮点分割\n" # 输出文本到 QTextBrowser
            self.ui.textBrowser_2.insertPlainText(outText1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            param_dict = torch.load('models/highlight.pth', map_location=torch.device('cpu'))
            new_param_dict = OrderedDict()
            for k, v in param_dict.items():
                name = k[7:]  # remove "module."
                new_param_dict[name] = v
            model = Resnet34_Unet(3, 1).to(device)
            model.load_state_dict(new_param_dict)
            model.eval()  # 预测模式
            with torch.no_grad():
                for index in range(len(self.imgPath)):
                    img = io.imread(self.imgPath[index])  # numpy.ndarray
                    self.img8bit = transfer_16bit_to_8bit(img)
                    img_pil = Image.fromarray(self.img8bit)
                    img_pil = img_pil.convert('RGB')
                    img_tensor = transforms_(img_pil)
                    img = torch.unsqueeze(img_tensor, 0).to(device)
                    output = model(img)
                    output = output.squeeze().detach().numpy()
                    output = np.array(output * 255, dtype='uint8')
                    _, binary_image = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    dirname = self.root + '-predict'
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    filename = dirname + '\\' + self.imgPath[index].split('\\')[-1]
                    # cv2.imwrite(filename, output)
                    cv2.imencode(filename[-4:], binary_image)[1].tofile(filename)
                    outText1_ = "保存图像：" + filename + "\n"
                    self.ui.textBrowser_2.insertPlainText(outText1_)
            pass

class MyClass(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyClass, self).__init__(parent)
        self.setupUi(self)
        self.selectPic.clicked.connect(self.openimage)
        self.segPic.clicked.connect(self.on_algorithm_changed)
        self.comboBox.currentIndexChanged.connect(self.algorithm_changed)
        self.savePic.clicked.connect(self.save_image)
        self.resetButton.clicked.connect(self.reset)
        self.text_browser = self.textBrowser

        self.batchPic.clicked.connect(self.open_child_window)

    def open_child_window(self):
        child_window = ChildWindow(self)
        child_window.show()

    def reset(self): # 重置所有相关的状态和控件
        self.lineEdit.clear()
        self.comboBox.setCurrentIndex(0)  # 重置算法选择
        self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                    "border: 1px solid rgb(224, 130, 43);")
        self.lineEdit.setEnabled(True)
        self.showImage_1.clear()  # 清除第一个图像显示区域
        self.showImage_2.clear()  # 清除第二个图像显示区域
        self.text_browser.clear()   # 清空 QTextBrowser 中的内容

    def algorithm_changed(self): # 根据选择的算法切换输入框的可编辑状态
        selected_algorithm = self.comboBox.currentText()
        if selected_algorithm == '阈值分割':  # 根据选择的算法执行相应的操作
            self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                        "border: 1px solid rgb(224, 130, 43);")
            self.lineEdit.setEnabled(True)
            pass
        else:
            self.lineEdit.setStyleSheet("background-color: rgb(239, 239, 239);\n"
                                        "border: 1px solid rgb(224, 130, 43);")
            self.lineEdit.setEnabled(False)
            self.lineEdit.clear()
            pass

    def openimage(self):
        self.imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "All Files(*);;*.jpg;;*.png;;*.tif")
        path, file_with_extension = os.path.split(self.imgName)
        file_name, extension = os.path.splitext(file_with_extension)
        new_file_name = file_name + '_seg' + extension  # 修改文件名为新的名称
        self.new_file_path = os.path.join(path, new_file_name)  # 重新构建路径

        self.image = transfer_16bit_to_8bit(io.imread(self.imgName))  # 读取图像
        self.height, self.width = self.image.shape
        qimg = QImage(self.image.data, self.width, self.height, QImage.Format_Grayscale8)   # 将图像转换为 PyQt5 图像
        pixmap = QPixmap.fromImage(qimg).scaled(self.showImage_1.width(), self.showImage_1.height())
        self.showImage_1.setPixmap(pixmap)
        outText1 = "输入图像：" + self.imgName + "\n"  # 输出文本到 QTextBrowser
        self.text_browser.insertPlainText(outText1)

    def on_algorithm_changed(self): # 处理算法选择变化的事件
        selected_algorithm = self.comboBox.currentText()
        if selected_algorithm == '阈值分割':  # 根据选择的算法执行相应的操作
            outText2 = "执行阈值分割 阈值设置为" + self.lineEdit.text() + "\n"  # 输出文本到 QTextBrowser
            self.text_browser.insertPlainText(outText2)
            threshold_value = int(self.lineEdit.text())
            _, binary_image = cv2.threshold(self.image, threshold_value, 255, cv2.THRESH_BINARY)  # 对图像进行二值化
            qimg = QImage(binary_image.data, self.width, self.height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg).scaled(self.showImage_1.width(), self.showImage_1.height())
            self.showImage_2.setPixmap(pixmap)
            self.ouputImg = binary_image
            pass

        elif selected_algorithm == 'OTSU分割':  # 根据选择的算法执行相应的操作
            outText2 = "执行OTSU分割\n"
            self.text_browser.insertPlainText(outText2)  # 输出文本到 QTextBrowser
            _, thresholded_image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 使用大律法（Otsu's method）找到合适的阈值
            qimg = QImage(thresholded_image.data, self.width, self.height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg).scaled(self.showImage_2.width(), self.showImage_2.height())
            self.showImage_2.setPixmap(pixmap)
            self.ouputImg = thresholded_image
            pass

        elif selected_algorithm == '亮点分割':  # 调用亮点分割模型
            # 输出文本到 QTextBrowser
            outText2 = "执行亮点分割\n"
            self.text_browser.insertPlainText(outText2)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            param_dict = torch.load('models/highlight.pth', map_location=torch.device('cpu'))
            new_param_dict = OrderedDict()
            for k, v in param_dict.items():
                name = k[7:]  # remove "module."
                new_param_dict[name] = v
            model = Resnet34_Unet(3, 1).to(device)
            model.load_state_dict(new_param_dict)
            model.eval()  # 预测模式
            img_pil = Image.fromarray(self.image)
            img_pil = img_pil.convert('RGB')
            img_tensor = transforms_(img_pil)
            img = torch.unsqueeze(img_tensor, 0).to(device)
            output = model(img)
            output = output.squeeze().detach().numpy()
            output = np.array(output * 255, dtype='uint8')
            _, thresholded_image = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            qimg = QImage(thresholded_image.data, self.width, self.height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg).scaled(self.showImage_2.width(), self.showImage_2.height())
            self.showImage_2.setPixmap(pixmap)
            self.ouputImg = thresholded_image
            pass

    def save_image(self):
        # 输出文本到 QTextBrowser
        outText3 = "保存图像：" + self.new_file_path + "\n"
        self.text_browser.insertPlainText(outText3)
        cv2.imwrite(self.new_file_path, self.ouputImg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyClass()
    myWin.show()
    sys.exit(app.exec_())
