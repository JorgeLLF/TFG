
import os
import json
import numpy as np

# ORIGINAL
# import Code.config as config
# from Code.utils.utils import transform_frames_to_time, transform_seconds_to_time, get_video_fps, transform_frames_to_seconds
# from Code.intervener_recognition.intervener_recognisers import FacialIntervenerRecogniser, VocalIntervenerRecogniser
# from Code.data_collecting.embeddings.embedding_file_savers import NPArraySerializer
# from Code.data_collecting.embeddings.embedding_file_readers import FacialEmbeddingsJSONReader, VocalEmbeddingsJSONReader
# from Code.data_collecting.video_audio_tracks.video_audio_diarizer import VoiceActivityDetector
# from Code.data_collecting.video_audio_tracks.video_audio_savers import VideoAudioExtractor

# PARA VSCODE
import sys
import os
sys.path.append(os.getcwd() + "/Code")
import config
from utils.utils import transform_frames_to_time, transform_seconds_to_time, get_video_fps, transform_frames_to_seconds
from intervener_recognition.intervener_recognisers import FacialIntervenerRecogniser, VocalIntervenerRecogniser
from data_collecting.embeddings.embedding_file_savers import NPArraySerializer
from data_collecting.embeddings.embedding_file_readers import FacialEmbeddingsJSONReader, VocalEmbeddingsJSONReader
from data_collecting.video_audio_tracks.video_audio_diarizer import VoiceActivityDetector
from data_collecting.video_audio_tracks.video_audio_savers import VideoAudioExtractor


# RECONOCIMIENTO FACIAL

# RUTA ORIGINAL
# video_path = "../../Data/Raw/Videos/ShotBoundaryVideos/231025_01 - Trim_01.mp4"

# RUTA PARA VSCODE
video_path = config.DATA_PATH + "/Raw/Videos/ShotBoundaryVideos/231025_01 - Trim_01.mp4"

shot_boundary_detections = np.array(
    [0, 48, 99, 296, 353, 528, 580, 677, 750])  # El 750 lo añadí yo manualmente, es la duración del video en frames
shot_types = ["PlanoMediano", "PlanoMediano", "PlanoMediano", "PlanoMediano", "PlanoCercano", "PlanoMediano",
              "PlanoCercano", "PlanoMediano"]


facial_intervener_recogniser = FacialIntervenerRecogniser()
json_pos_facial_interveners = []
for i in range(1, len(shot_boundary_detections)):
    json_pos_facial_interveners.append(facial_intervener_recogniser.recognise(video_path,
                                                                     shot_boundary_detections[i - 1],
                                                                     shot_boundary_detections[i],
                                                                     shot_types[i - 1]))
print("")
print(json_pos_facial_interveners)
print("")

facial_embeddings_file_content = FacialEmbeddingsJSONReader().read_embeddings_from_file()
facial_interveners_ids = []
for intervener in json_pos_facial_interveners:
    if isinstance(intervener, str):
        print(intervener)
        facial_interveners_ids.append(intervener) # Estamos pegando "Diputado desconocido"
    else:
        print(
            facial_embeddings_file_content["id"][intervener] + " " + facial_embeddings_file_content["file"][intervener])
        facial_interveners_ids.append(facial_embeddings_file_content["id"][intervener])
print("")
print("")


# Tener un directorio en Processed que se llame Interventions. Dentro de este, un directorio con el nombre del video,
# y dentro de este directorio 2 json: uno con las intervenciones faciales y otro con las intervenciones vocales.

video_name = video_path.split("/")[-1].split(".")[0]
if not os.path.exists(config.FACIAL_INTERVENTIONS_PROCESSED_DATA_PATH):
    os.makedirs(config.FACIAL_INTERVENTIONS_PROCESSED_DATA_PATH)

facial_interventions_file_content = {"begin": shot_boundary_detections[:-1],
                                     "type": shot_types,
                                     "intervener": facial_interveners_ids}
with open(os.path.join(config.FACIAL_INTERVENTIONS_PROCESSED_DATA_PATH,
                       "facial_interventions_{}.json".format(video_name)), 'w') as facial_interventions_json_file:
    json.dump(facial_interventions_file_content, facial_interventions_json_file, cls=NPArraySerializer)



# RECONOCIMIENTO VOCAL

video_path = config.DATA_PATH + "/Raw/Videos/ShotBoundaryVideos/231025_01 - Trim_01.mp4"
video_audio_extractor = VideoAudioExtractor()
video_audio_extractor.extract_full_audio(video_path)


video_name = video_path.split("/")[-1].split(".")[0]
audio = config.DATA_PATH + "/Processed/vocal_recognition/VideoAudios/FullAudios/" + video_name + ".wav"
audio_diarization = VoiceActivityDetector().diarize_voice_activity(audio=audio)
# print("")
# print(diarization)


# audio_time_in_frames = 10*fps
json_pos_vocal_interveners = []
vocal_interveners_ids = []
vocal_intervener_recogniser = VocalIntervenerRecogniser()
for intervention in audio_diarization:
    json_pos_vocal_interveners.append(vocal_intervener_recogniser.recognise(audio,
                                                                   intervention[0],
                                                                   intervention[1]))
# vad.write_vad_to_file(video=video, output_file_path="output.rttm")

print("")
print(json_pos_vocal_interveners)
print("")

vocal_embeddings_file_content = VocalEmbeddingsJSONReader().read_embeddings_from_file()
for intervener in json_pos_vocal_interveners:
    if isinstance(intervener, int):
        print("Intervención muy corta")
    else:
        print(
            vocal_embeddings_file_content["id"][intervener] + " " + vocal_embeddings_file_content["file"][intervener])
        vocal_interveners_ids.append(vocal_embeddings_file_content["id"][intervener])
print("")
print("")

video_name = video_path.split("/")[-1].split(".")[0]
if not os.path.exists(config.VOCAL_INTERVENTIONS_PROCESSED_DATA_PATH):
    os.makedirs(config.VOCAL_INTERVENTIONS_PROCESSED_DATA_PATH)

vocal_interventions_file_content = {"begin": [intervention[0] for intervention in audio_diarization if intervention[1] - intervention[0] > 5],
                                    "finish": [intervention[1] for intervention in audio_diarization if intervention[1] - intervention[0] > 5],
                                    "intervener": vocal_interveners_ids}
with open(os.path.join(config.VOCAL_INTERVENTIONS_PROCESSED_DATA_PATH,
                       "vocal_interventions_{}.json".format(video_name)), 'w') as vocal_interventions_json_file:
    json.dump(vocal_interventions_file_content, vocal_interventions_json_file, cls=NPArraySerializer)