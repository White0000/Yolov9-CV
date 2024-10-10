import cv2
import pytesseract

# 使用Tesseract OCR从图像中提取文本
def extract_text_from_image(image):
    # 预处理图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 使用Tesseract进行文本识别
    recognized_text = pytesseract.image_to_string(binary_image, lang='chi_sim+eng')
    return recognized_text

# 测试OCR识别
if __name__ == "__main__":
    img = cv2.imread('/dataset/IIIT5K/test/')  # 替换为你要识别的图像路径
    text = extract_text_from_image(img)
    print(f'识别到的文本: {text}')
