import cv2

cap = cv2.VideoCapture(0)  # LIVE CAMERA

def get_frame():
    success, frame = cap.read()
    if not success:
        return None
    return frame
