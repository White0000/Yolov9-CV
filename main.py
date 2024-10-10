import sys
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QTextEdit, QFileDialog, QVBoxLayout, \
    QProgressBar, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, QObject
import pytesseract
from datetime import datetime
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
import torch
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# 定义信号类，用于线程间通信
class WorkerSignals(QObject):
    result = pyqtSignal(str)  # 用于传递识别结果
    progress = pyqtSignal(int)  # 用于传递进度条更新
    error = pyqtSignal(str)  # 用于传递错误信息


class TextRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # EAST 模型路径
        self.east_model_path = 'models/frozen_east_text_detection.pb'
        self.east_boxes = None
        self.show_boxes = False  # 控制框显隐的标志位

        # 实例化信号类
        self.signals = WorkerSignals()
        self.signals.result.connect(self.show_result)  # 连接识别结果信号
        self.signals.progress.connect(self.update_progress)  # 连接进度条信号
        self.signals.error.connect(self.show_error_dialog)  # 连接错误信号

    def initUI(self):
        self.setWindowTitle("图片文字识别系统")

        self.label = QLabel(self)
        self.label.setText("请选择一张图片")

        self.button = QPushButton('选择图片', self)
        self.button.clicked.connect(self.open_image)

        self.grayscale_button = QPushButton('灰度化', self)
        self.grayscale_button.clicked.connect(self.apply_grayscale)
        self.grayscale_button.setEnabled(False)

        self.binarize_button = QPushButton('二值化', self)
        self.binarize_button.clicked.connect(self.apply_binarization)
        self.binarize_button.setEnabled(False)

        self.denoise_button = QPushButton('去噪', self)
        self.denoise_button.clicked.connect(self.apply_denoise)
        self.denoise_button.setEnabled(False)

        self.select_button = QPushButton('选择识别区域', self)
        self.select_button.clicked.connect(self.select_area)
        self.select_button.setEnabled(False)

        self.process_button = QPushButton('开始识别', self)
        self.process_button.clicked.connect(self.start_recognition)
        self.process_button.setEnabled(False)

        self.east_button = QPushButton('EAST模型检测文字框', self)
        self.east_button.clicked.connect(self.toggle_east_boxes)
        self.east_button.setEnabled(False)  # 初始禁用，加载图片后启用

        self.result_text = QTextEdit(self)

        self.export_button = QPushButton('导出为TXT', self)
        self.export_button.clicked.connect(self.export_result)
        self.export_button.setEnabled(False)  # 初始禁用，识别完成后启用

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.grayscale_button)
        layout.addWidget(self.binarize_button)
        layout.addWidget(self.denoise_button)
        layout.addWidget(self.select_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.east_button)
        layout.addWidget(self.result_text)
        layout.addWidget(self.export_button)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        self.image_path = None
        self.image = None
        self.selected_roi = None
        self.recognized_text = ""
        self.is_grayscale = False
        self.original_image = None  # 保存原始图像

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                                   "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            self.image = cv2.imread(file_name)
            self.original_image = self.image.copy()  # 保存原始彩色图像
            if self.image is not None:
                self.display_image(self.image)
                self.grayscale_button.setEnabled(True)
                self.east_button.setEnabled(True)  # 图片加载后启用 EAST 按钮
                self.select_button.setEnabled(True)  # 启用选择识别区域按钮
            else:
                self.result_text.setText("无法读取图片，请选择有效的图片文件")

    def display_image(self, img):
        if len(img.shape) == 3:  # 彩色图像
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            qImg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        elif len(img.shape) == 2:  # 灰度图像
            height, width = img.shape
            bytes_per_line = width
            qImg = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            self.show_error_dialog("无法显示图像，图像通道数不正确")
            return
        self.label.setPixmap(QPixmap.fromImage(qImg))

    def apply_grayscale(self):
        if self.image is not None:
            if len(self.image.shape) == 3:  # 确保图像是彩色图像时进行灰度化
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                self.is_grayscale = True
                self.display_image(self.image)
                self.grayscale_button.setEnabled(False)
                self.binarize_button.setEnabled(True)
                self.process_button.setEnabled(True)  # 启用识别按钮，允许随时识别
            else:
                self.result_text.setText("图像已经是灰度图。")
        else:
            self.result_text.setText("请先选择图片")

    def apply_binarization(self):
        if self.image is not None:
            self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
            self.display_image(self.image)
            self.binarize_button.setEnabled(False)
            self.denoise_button.setEnabled(True)
        else:
            self.result_text.setText("请先选择图片")

    def apply_denoise(self):
        if self.image is not None:
            self.image = cv2.fastNlMeansDenoising(self.image, None, 30, 7, 21)
            self.display_image(self.image)
            self.denoise_button.setEnabled(False)
            self.select_button.setEnabled(True)
        else:
            self.result_text.setText("请先选择图片")

    def select_area(self):
        if self.image is not None:
            try:
                img_to_select = self.image if not self.is_grayscale else cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
                roi = cv2.selectROI("选择识别区域", img_to_select, showCrosshair=True)

                x, y, w, h = roi
                if w > 0 and h > 0:
                    self.selected_roi = self.image[y:y + h, x:x + w]
                    cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用绿色框选区域
                    self.display_image(self.image)
                    self.process_button.setEnabled(True)  # 启用识别按钮
                else:
                    self.show_error_dialog("选择的区域无效，请重新选择一个有效的区域。")
            except Exception as e:
                self.show_error_dialog(f"发生错误: {str(e)}")
            finally:
                cv2.destroyAllWindows()
        else:
            self.result_text.setText("请先选择图片")

    def start_recognition(self):
        # 在子线程中执行识别任务，避免主线程卡顿
        recognition_thread = threading.Thread(target=self.process_image)
        recognition_thread.start()

    def process_image(self):
        if self.selected_roi is not None:
            self.signals.progress.emit(10)  # 更新进度

            try:
                # 使用 Tesseract 进行 OCR 识别
                recognized_text = pytesseract.image_to_string(self.selected_roi, lang='chi_sim+eng')

                # 处理识别结果并使用 LayoutLMv1 进一步优化
                self.process_with_layoutlm(recognized_text)

                self.signals.progress.emit(100)  # 更新进度
                self.export_button.setEnabled(True)  # 启用导出按钮
            except Exception as e:
                self.signals.error.emit(f"识别时出现错误: {str(e)}")

        else:
            self.signals.error.emit("请先选择区域")

    def process_with_layoutlm(self, recognized_text):
        # 使用 LayoutLMv1 来处理 OCR 结果的文本布局
        tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

        # Tokenization
        words = recognized_text.split()
        encoding = tokenizer(words, is_split_into_words=True, return_tensors="pt")

        # 获取模型输出
        outputs = model(**encoding)
        logits = outputs.logits
        print("Logits from LayoutLMv1: ", logits)

        # 更新识别结果显示
        self.signals.result.emit(recognized_text)  # 在主线程中显示 LayoutLM 处理后的结果

    def toggle_east_boxes(self):
        if self.east_boxes is None:
            self.detect_text_east()  # 如果还没有检测过，先进行文本检测
        else:
            self.show_boxes = not self.show_boxes  # 切换显隐状态
            self.update_east_display()

    def detect_text_east(self):
        net = cv2.dnn.readNet(self.east_model_path)

        orig_image = self.original_image.copy()  # 使用原始彩色图像
        h, w = orig_image.shape[:2]

        # EAST 模型的输入尺寸必须是32的倍数
        new_w, new_h = (w // 32) * 32, (h // 32) * 32
        resized_image = cv2.resize(orig_image, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), swapRB=True,
                                     crop=False)
        net.setInput(blob)

        # 获取输出层
        (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
        self.east_boxes = self.decode_predictions(scores, geometry, 0.5)

        # 恢复到原图尺寸的比例
        ratio_w, ratio_h = w / float(new_w), h / float(new_h)
        for (start_x, start_y, end_x, end_y) in self.east_boxes:
            start_x = int(start_x * ratio_w)
            start_y = int(start_y * ratio_h)
            end_x = int(end_x * ratio_w)
            end_y = int(end_y * ratio_h)
            cv2.rectangle(orig_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        self.image = orig_image
        self.show_boxes = True  # 默认显示检测框
        self.update_east_display()

    def update_east_display(self):
        if self.show_boxes:
            self.display_image(self.image)  # 显示有框的图像
        else:
            img_no_boxes = cv2.imread(self.image_path)  # 还原到原始图片，无框显示
            self.display_image(img_no_boxes)

    def decode_predictions(self, scores, geometry, min_confidence):
        (num_rows, num_cols) = scores.shape[2:4]
        boxes = []
        confidences = []

        for y in range(0, num_rows):
            scores_data = scores[0, 0, y]
            x_data0 = geometry[0, 0, y]
            x_data1 = geometry[0, 1, y]
            x_data2 = geometry[0, 2, y]
            x_data3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(0, num_cols):
                if scores_data[x] < min_confidence:
                    continue

                # 计算偏移量
                (offset_x, offset_y) = (x * 4.0, y * 4.0)

                angle = angles_data[x]
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]

                end_x = int(offset_x + (cos_a * x_data1[x]) + (sin_a * x_data2[x]))
                end_y = int(offset_y - (sin_a * x_data1[x]) + (cos_a * x_data2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)

                boxes.append((start_x, start_y, end_x, end_y))
                confidences.append(float(scores_data[x]))

        return boxes

    def show_result(self, result):
        self.result_text.setText(result)  # 在主线程中显示识别结果
        self.recognized_text = result  # 保存识别结果

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def export_result(self):
        if self.recognized_text.strip():  # 检查是否有有效的识别结果
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"识别结果_{current_time}.txt"
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "导出为TXT", file_name,
                                                       "Text Files (*.txt);;All Files (*)", options=options)
            if file_name:
                with open(file_name, 'w', encoding='utf-8') as file:
                    file.write(self.recognized_text)
        else:
            self.result_text.setText("没有识别结果可以导出")

    def show_error_dialog(self, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("错误")
        error_dialog.setText(message)
        error_dialog.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TextRecognitionApp()
    window.show()
    sys.exit(app.exec_())
