
import os
import json
import cv2


def get_specific_frame(video_path, frame_num):
    cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print("Error al abrir el video.")
    #     return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    # cv.imshow("frame", frame)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    cap.release()
    if ret:
        return frame
    else:
        print(f"No se pudo obtener el frame.")
        return None


def load_model_classes_from_file(classes_file):
    with open(classes_file, 'r') as file:
        classes = json.load(file)
        return classes


def transform_frames_to_time(frames, fps):
    total_seconds = frames / fps
    return transform_seconds_to_time(total_seconds)

def transform_seconds_to_time(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = int(seconds % 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)

def transform_frames_to_seconds(frames, fps):
    spf = fps**(-1)
    seconds = frames * spf
    return seconds


def count_files(path):
    return len(os.listdir(os.path.abspath(path)))


def get_video_fps(video):
    cap = cv2.VideoCapture(os.path.abspath(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_video_frames(video):
    cap = cv2.VideoCapture(os.path.abspath(video))
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames

def get_video_seconds(video):
    fps = get_video_fps(video)
    frames = get_video_frames(video)

    return frames * (1/fps)