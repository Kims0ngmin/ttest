import cv2
import numpy as np
import sys

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 128, 255), -1, lineType=cv2.LINE_AA)

    return vis

cap = cv2.VideoCapture("video1.mp4")

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (540, 380))

if not ret:
    print('frame read failed!')
    sys.exit()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

threshold = 0.5  # 임계값 설정

frame_counter = 0

while True:
    ret, frame2 = cap.read()
    frame2 = cv2.resize(frame2, (540, 380))

    if not ret:
        print('frame read failed!')
        sys.exit()

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 13, 3, 5, 1.1, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])


    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 모션 벡터 평균 계산
    avg_mag = np.mean(mag)
    print(avg_mag)

    # 임계값을 초과하는 모션 감지
    if avg_mag > threshold:
        save_path = f"./save/motion_detected_{frame_counter}.jpg"
        cv2.imwrite(save_path, frame2)
        print(f"Motion detected! Frame saved as {save_path}")
        frame_counter += 1

    cv2.imshow('frame', frame2)
    cv2.imshow('flow', bgr)
    cv2.imshow('frame2', draw_flow(gray2, flow))
    if cv2.waitKey(20) == 27:
        break

    gray1 = gray2

cap.release()
cv2.destroyAllWindows()
