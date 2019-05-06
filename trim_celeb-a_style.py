# Webカメラで撮影した写真を、Celeb-Aと同じ形（全体のトリミング、目の位置調整）で、保存するツール
# 使用するときは、cv2.rectangle()、cv2.circle()の行をコメントアウトする

import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.imshow('img', img)

    #sが押されたらファイル保存、qが押されたら終了
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        path = "photo.jpg"
        cv2.imwrite(path, img)

cap.release()
cv2.destroyAllWindows()


# cascadeファイルをカレントに置いておく。対象は下記の２つのファイル
# 次のURLからダウンロードする。https://github.com/opencv/opencv/tree/master/data/haarcascades

face_cascade_path = './cascade/haarcascade_frontalface_default.xml'
eye_cascade_path = './cascade/haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)

src = cv2.imread('./photo.jpg')
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(src_gray)

print('faces : ',faces)

for x, y, w, h in faces:
    cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = src[y: y + h, x: x + w]
    face_gray = src_gray[y: y + h, x: x + w]
#    eyes = np.array(eye_cascade.detectMultiScale(face_gray))
    eyes = eye_cascade.detectMultiScale(face_gray)
    print('eyes : ', eyes)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        print('ex : ', ex, 'ey : ', ey, 'ew : ', ew, 'eh : ', eh)

    # eyes[0,0] : 目のxの位置
    # eyes[0,1] : 目のyの位置
    # eyes[0,2] : 目の幅
    # eyes[0,3] : 目の高さ

#目の中心座標の取得
eye0_x_cent = int(x + int(eyes[0][0]+eyes[0][2]/2))
eye1_x_cent = int(x + int(eyes[1][0]+eyes[1][2]/2))
eye1_y_cent = int(y + int(eyes[1][1]+eyes[1][3]/2))

print("eye0_x_cent : ",eye0_x_cent)

# 目の位置に目印の円を描く
cv2.circle(src, (x + eyes[0][0], y + eyes[0][1]), 10, (255,0,0), -1)
cv2.circle(src, (x + eyes[1][0], y + eyes[1][1]), 10, (0,255,0), -1)
cv2.circle(src, (eye1_x_cent, eye1_y_cent), 10, (0,0,255), -1)
cv2.circle(src, (eye0_x_cent, eye1_y_cent), 10, (255,255,0), -1)

#目の間の距離の取得と顔の中心の取得
dist_eyes = abs(eye1_x_cent - eye0_x_cent) # distance of eyes
center_x_face = int(abs(eye1_x_cent + eye0_x_cent) / 2)
center_y_face = eye1_y_cent

cv2.circle(src, (int(abs(eye1_x_cent + eye0_x_cent) / 2), eye1_y_cent), 10, (0,0,0), -1) # center of the face

# celebAの切り出し位置に赤の四角形の描画
cv2.rectangle(src, (int(center_x_face - 2.5 * dist_eyes), center_y_face - 3 * dist_eyes),\
 (int(center_x_face + 2.5 * dist_eyes), center_y_face + 3 * dist_eyes), (0,0,255), 2)

#切り出し処理
clipped_img = src[center_y_face - 3 * dist_eyes:center_y_face + 3 * dist_eyes,\
int(center_x_face - 2.5 * dist_eyes):int(center_x_face + 2.5 * dist_eyes) :] # yとxが逆

print('x, y, w, h', x, y, w, h)

# cv2.imwrite('./opencv_face_detect.jpg', src)
cv2.imwrite('./clipped.jpg', clipped_img)
