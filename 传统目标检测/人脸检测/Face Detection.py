# 作者：水果好好吃哦
# 日期：2023/7/18
import cv2
import tkinter as tk
from tkinter import filedialog


def img_test():
    # 获取选择文件路径
    # 实例化
    root = tk.Tk()
    root.withdraw()
    # 获取文件或文件夹的绝对路径路径
    return filedialog.askopenfilename()


def haar_detection():
    face_path = 'haarcascades\\haarcascade_frontalface_default.xml'
    eye_path = 'haarcascades\\haarcascade_eye.xml'
    smile_path = 'haarcascades\\haarcascade_smile.xml'
    diction = {'face': face_path, 'eye': eye_path, 'smile': smile_path}
    for i in diction:
        diction[i] = cv2.CascadeClassifier(diction[i])
    return diction


def image_detection(img, detection):
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    face_ret = detection['face'].detectMultiScale(image_gray, scaleFactor=1.02, minNeighbors=5, minSize=(15, 15),
                                                  maxSize=(50, 50), flags=cv2.CASCADE_DO_CANNY_PRUNING)
    if len(face_ret) != 0:
        for (x, y, w, h) in face_ret:
            face_roi = image_gray[y:y + h, x:x + w]
            eye_ret = detection['eye'].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)
            smile_ret = detection['smile'].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15),
                                                            flags=cv2.CASCADE_SCALE_IMAGE)
            for (xx, yy, ww, hh) in eye_ret:
                pt1 = (x + xx, y + yy)
                pt2 = (pt1[0]+ww, pt1[1]+hh)
                cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)
            for (xx, yy, ww, hh) in smile_ret:
                pt1 = (x + xx, y + yy)
                pt2 = (pt1[0]+ww, pt1[1]+hh)
                cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print('图像中有人脸')
    else:
        print('图像中无人脸')

    return img


if __name__ == '__main__':
    image = cv2.imread('image\\icon.jpg')
    cv2.namedWindow('Press q to exit and n to load the next picture', cv2.WINDOW_NORMAL)
    cv2.imshow('Press q to exit and n to load the next picture', image)
    while 1:
        k = cv2.waitKey()
        if k == ord('n'):
            image = cv2.imread(img_test())
            detector = haar_detection()
            image = image_detection(image, detector)
            cv2.imshow('Press q to exit and n to load the next picture', image)
        elif k == ord('q'):
            break
    cv2.destroyAllWindows()
