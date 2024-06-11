
# import numpy as np
# from code.ShotBoundaryDetection.ColorHistogramShotBoundaryDetector import ColorHistogramShotBoundaryDetector
# from datetime import timedelta
import torch
from torch.nn.functional import one_hot

import sys
import os
import json
import bisect
import statistics
import pandas as pd

# sys.path.append(os.getcwd() + "/Code")
# import config as config
# from diarization_data_preprocessing.diarization_preprocessor import DiarizationPreprocessor
# from data_collecting.video_audio_tracks.video_audio_diarizer import VoiceActivityDetector



# PRUEBA 1

# video_list = ["../Data/Raw/Videos/ShotBoundaryVideos/231025_01 - Trim_01.mp4"]
# detector = ColorHistogramShotBoundaryDetector(video_list)
# videos = {}
# for video in video_list:
#     videos[video] = detector.detect_shot_boundaries(video)
#
# print(videos)
#
# for video in videos.keys():
#     print("{}".format(video))
#     detector_output = videos[video]
#     predictions = detector_output[0]
#     video_fps = detector_output[1]
#     for shot in predictions:
#         print("Comienzo de escena: {}. Final de escena: {}.".format(
#             timedelta(round(shot[0] * (1 / video_fps), 2)),
#             timedelta(round(shot[1] * (1 / video_fps), 2))))


# PRUEBA 2

# array_prueba = np.array([1, 2, 3, 4])
# print(array_prueba + 1)

# ruta_prueba = os.path.abspath("transnetv2/inference-pytorch")
# print(ruta_prueba)
# if ruta_prueba in sys.path:
#     print("Está en los módulos de búsqueda")
# else:
#     print("No está en los módulos de búsqueda")


# PRUEBA 3
# tensor = torch.tensor([0,1,2,3])
# print(one_hot(tensor))




# PRUEBA 4

# def get_indices_included(values, min_value, max_value):
#     begin_indices_included = bisect.bisect_right(values, min_value)
#     finish_indices_included = bisect.bisect_left(values, max_value)
#     return begin_indices_included, finish_indices_included

# def get_associated_name(df, id):
#         # print(df)
#         # print(id)
#         # print(df[df["ID"] == id])
#         return df[df["ID"] == id].iloc[0]["Nombre"]


# # CARGA DE DATOS
# video_name = "PruebaLarga"
# with open(os.path.join(config.FACIAL_INTERVENTIONS_PROCESSED_DATA_PATH, 
#                                "facial_interventions_{}.json".format(video_name)), 'r') as facial_interventions_json_file, \
#              open(os.path.join(config.VOCAL_INTERVENTIONS_PROCESSED_DATA_PATH, 
#                                "vocal_interventions_{}.json".format(video_name)), 'r') as vocal_interventions_json_file:

#     facial_interventions_content = json.load(facial_interventions_json_file)
#     vocal_interventions_content = json.load(vocal_interventions_json_file)


# # DATOS PARA PROCESO DE CONTEO
# facial_begin_content = facial_interventions_content["begin"]
# facial_intervener_content = facial_interventions_content["intervener"]

# total_interventions = len(vocal_interventions_content["intervener"])
# actual_vocal_intervener = vocal_interventions_content["intervener"][0]
# actual_vocal_begin = vocal_interventions_content["begin"][0]
# vocal_checked_interventions = 1
# facial_checked_interventions = 0


# # METEREMOS AQUÍ TUPLAS CON EL PRINCIPIO Y FINAL DE INTERVENCIONES VOCALES Y EL DIPUTADO
# diarization_file_content = []
# deputies_names_df = pd.read_csv(config.RAW_DATA_PATH + "/MiembrosPleno/deputies_names.csv", encoding="utf-8", dtype={"ID":str})
# print("")

# # LÓGICA DE COMPROBACIÓN DE DIPUTADOS FACIAL Y VOCAL POR INTERVALOS DE TIEMPO
# while vocal_checked_interventions < total_interventions:

#         # Como se repiten mucho los diputados de forma seguida por los mini silencios, vamos a recorrer todas sus intervenciones seguidas
#         if vocal_interventions_content["intervener"][vocal_checked_interventions] != actual_vocal_intervener:
                        
#                 last_intervention = vocal_checked_interventions - 1 # EL ANTERIOR, es la última vez que aparece
#                 actual_vocal_finish = vocal_interventions_content["finish"][last_intervention]
#                 facial_begin_indices_included, facial_finish_indices_included = get_indices_included(
#                         facial_begin_content, actual_vocal_begin, actual_vocal_finish)

#                 # TODO : Contar el número de veces que aparece cada diputado facialmente, escoger al que más aparece y comparar con el vocal
#                 facial_interveners_included = facial_intervener_content[facial_begin_indices_included:facial_finish_indices_included]
#                 print("Comienzo y final de intervención vocal: {} - {}".format(actual_vocal_begin, actual_vocal_finish))
#                 print("Comienzo y final de intervención facial: {} - {}".format(facial_begin_content[facial_begin_indices_included], facial_begin_content[facial_finish_indices_included]))
#                 print("Intervinientes faciales incluidos: {}".format(facial_interveners_included))
#                 print("Interviniente vocal: {}".format(actual_vocal_intervener))


#                 # Obtención de diccionario de tiempo por diputado
#                 facial_begin_included = facial_begin_content[facial_begin_indices_included:facial_finish_indices_included + 1]
#                 deputy_time_dict = {}
#                 # print(len(facial_interveners_included))
#                 # print(len(facial_begin_included))

#                 for i, deputy in enumerate(facial_interveners_included):
#                         if deputy not in deputy_time_dict:
#                               if deputy != "Diputado desconocido":
#                                 deputy_time_dict[deputy] = facial_begin_included[i+1] - facial_begin_included[i]
#                         else:
#                               deputy_time_dict[deputy] += facial_begin_included[i+1] - facial_begin_included[i]
                
#                 if len(deputy_time_dict.keys()) > 0:
#                         facial_intervener_mode = max(deputy_time_dict, key=deputy_time_dict.get)
#                         print("Moda facial: {}".format(facial_intervener_mode))

#                         # Si el interviniente facial y vocal son distintos, ponemos desconocido
#                         if actual_vocal_intervener != facial_intervener_mode:
#                                 diarization_file_content.append((actual_vocal_begin, actual_vocal_finish, "Diputado desconocido"))
#                                 # diarization_file_content.append((actual_vocal_begin, facial_begin_content[facial_finish_indices_included], "Diputado desconocido"))
#                         else:
#                                 diarization_file_content.append((actual_vocal_begin, actual_vocal_finish, get_associated_name(deputies_names_df, actual_vocal_intervener)))
#                                 # diarization_file_content.append((actual_vocal_begin, facial_begin_content[facial_finish_indices_included], get_associated_name(deputies_names_df, actual_vocal_intervener)))

#                 # actual_vocal_begin = actual_vocal_finish
#                 actual_vocal_begin = vocal_interventions_content["begin"][last_intervention+1]
#                 # actual_vocal_intervener = vocal_interventions_content["intervener"][vocal_checked_interventions]
#                 actual_vocal_intervener = vocal_interventions_content["intervener"][vocal_checked_interventions+1]

#                 facial_checked_interventions += facial_finish_indices_included - facial_begin_indices_included # Sumo la cantidad de indices ya comprobados

#                 print("Nuevo comienzo de vocal: {}".format(actual_vocal_begin))
#                 print("Nuevo diputado: {}".format(actual_vocal_intervener))
#                 print("Intervenciones faciales comprobadas: {}".format(facial_checked_interventions))
#                 print("")

#         vocal_checked_interventions += 1


# # TRATAMOS LA ÚLTIMA INTERVENCIÓN

# # print(actual_vocal_begin)
# # print(vocal_interventions_content["finish"][-1])
# # print(facial_begin_content)

# actual_vocal_finish = vocal_interventions_content["finish"][-1]
# facial_begin_indices_included, facial_finish_indices_included = get_indices_included(
#                         facial_begin_content, actual_vocal_begin, actual_vocal_finish)

# facial_interveners_included = facial_intervener_content[facial_begin_indices_included:facial_finish_indices_included]
# # facial_intervener_mode = statistics.mode(facial_interveners_included)

# facial_begin_included = facial_begin_content[facial_begin_indices_included:facial_finish_indices_included+1]
# print("Intervinientes faciales incluidos: {}".format(facial_begin_included))
# # print(len(facial_begin_included))
# # print(len(facial_interveners_included))
# # print(len(facial_begin_content))
# # print(len(facial_intervener_content))

# deputy_time_dict = {}
# for i, deputy in enumerate(facial_interveners_included):
#         if deputy not in deputy_time_dict:
#                 if deputy != "Diputado desconocido":
#                         deputy_time_dict[deputy] = facial_begin_included[i+1] - facial_begin_included[i]
#         else:
#                 deputy_time_dict[deputy] += facial_begin_included[i+1] - facial_begin_included[i]

# facial_intervener_mode = max(deputy_time_dict, key=deputy_time_dict.get)

# print("Numero final de intervenciones comprobadas {}".format(facial_checked_interventions))
# print("Interviniente vocal final: {}".format(actual_vocal_intervener))
# print("Interviniente facial final: {}".format(facial_intervener_mode))

# if actual_vocal_intervener != facial_intervener_mode:
#         diarization_file_content.append((actual_vocal_begin, actual_vocal_finish, "Diputado desconocido"))
# else:
#         diarization_file_content.append((actual_vocal_begin, actual_vocal_finish, get_associated_name(deputies_names_df, actual_vocal_intervener)))


# # RESULTADOS
# print("")
# print(diarization_file_content)
# print("")

# video_name = "PruebaLarga"
# diarization_preprocessor = DiarizationPreprocessor()
# print(diarization_preprocessor.preprocess(video_name))


# vad = VoiceActivityDetector()
# print(vad._pipeline.parameters()["segmentation"]["min_duration_off"])


###################


def funcion_con_intermedia(func_intermedia):
    # Sentencias comunes al principio
    print("Estas son las sentencias comunes al principio.")

    # Llamada a la función intermedia
    func_intermedia()

    # Sentencias comunes al final
    print("Estas son las sentencias comunes al final.")


# Función intermedia 1
def funcion_intermedia_1():
    print("Esta es la función intermedia 1.")


# Función intermedia 2
def funcion_intermedia_2():
    print("Esta es la función intermedia 2.")


# Llamar a la función con la función intermedia deseada
funcion_con_intermedia(funcion_intermedia_1)
funcion_con_intermedia(funcion_intermedia_2)
