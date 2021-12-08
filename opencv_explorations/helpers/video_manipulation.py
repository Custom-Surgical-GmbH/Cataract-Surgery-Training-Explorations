
import cv2


# frame_start inclusive, frame_end exclusive
def extract_consecutive(video_path, frame_start, frame_end, mode='bgr'):
    assert frame_end > frame_start >= 0
    assert mode in ('bgr', 'rgb', 'hsv', 'gray', 'h', 's', 'v')

    frames = []

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    while cap.isOpened():
        ret, base = cap.read()
        if not ret:
            break

        if frame_index == frame_end:
            break

        if frame_index >= frame_start:
            if mode == 'bgr':
                frames.append(base.copy())
            elif mode == 'rgb':
                frames.append(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
            elif mode == 'hsv':
                frames.append(cv2.cvtColor(base, cv2.COLOR_BGR2HSV))
            elif mode == 'gray':
                frames.append(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY))
            elif mode == 'h':
                frames.append(cv2.cvtColor(base, cv2.COLOR_BGR2HSV)[:, :, 0])
            elif mode == 's':
                frames.append(cv2.cvtColor(base, cv2.COLOR_BGR2HSV)[:, :, 1])
            elif mode == 'v':
                frames.append(cv2.cvtColor(base, cv2.COLOR_BGR2HSV)[:, :, 2])

        frame_index += 1

    cap.release()

    return frames
