
import datetime
import cv2
import numpy as np


import sys
import os
sys.path.append(os.getcwd() + "/Code")
from utils.utils import transform_frames_to_time
from data_collecting.video_audio_tracks.video_audio_diarizer import VoiceActivityDetector
from data_collecting.video_audio_tracks.video_audio_savers import VideoAudioExtractor


input_audio = "../../Data/Raw/231025_01 - Trim_01.mp4"
vad = VoiceActivityDetector()
diarization = vad.diarize_voice_activity(input_audio_path=input_audio)
print("")
print(diarization)
vad.write_vad_to_file(input_audio_path=input_audio, output_file_path="output.rttm")

seconds = 0.008488964346349746
formatted_time = str(datetime.timedelta(seconds=seconds)).split(".")[0]
if len(formatted_time) < 8:
    print("0" + formatted_time)
else:
    print(formatted_time)


input_video_path = "../../Data/Raw/Videos/ShotBoundaryVideos/231025_01 - Trim_01.mp4"
output_video_path = "MiembrosPleno/Pruebas"
# voice_audio_name = "voice_audio_0001.wav"
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

shot_boundary_detections = np.array([0, 48, 99, 296, 353, 528, 580, 677])
audio_time = 10 * fps # 10 segundos máximo por los fps

for i in range(1, len(shot_boundary_detections)):
    ss = transform_frames_to_time(shot_boundary_detections[i - 1], fps)
    if audio_time <= shot_boundary_detections[i] - shot_boundary_detections[i-1]: # Si el tiempo entre cambios de plano es mayor a 10 segundos, nos quedamos con un audio de 10 los 10 segundos iniciales
        to = transform_frames_to_time(shot_boundary_detections[i - 1] + audio_time, fps)
    else: # Si el tiempo entre cambios de plano es menor a 10 segundos, nos quedamos
        to = transform_frames_to_time(shot_boundary_detections[i], fps)

    VideoAudioExtractor().extract_audio(input_video_path=input_video_path, output_audio_folder_name=output_video_path,
                                        ss=ss, to=to)




