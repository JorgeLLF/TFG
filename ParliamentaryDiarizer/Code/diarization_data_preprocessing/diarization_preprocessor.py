
import sys
import os
import json
import bisect
import pandas as pd

sys.path.append(os.getcwd() + "/Code")
import Code.config as config



class DiarizationPreprocessor():

    def __init__(self):
        self._df = pd.read_csv(config.RAW_DATA_PATH + "/MiembrosPleno/deputies_names.csv", dtype={"ID":str})


    def _get_included_indices(self, values, min_value, max_value):
        begin_included_index = bisect.bisect(values, min_value)
        finish_included_index = bisect.bisect(values, max_value)
        return begin_included_index, finish_included_index

    def _get_associated_name(self, id):
        return self._df[self._df["ID"] == id].iloc[0]["Nombre"]
    

    def preprocess(self, video_name):
           
        # CARGA DE DATOS
        with open(os.path.join(config.FACIAL_INTERVENTIONS_PROCESSED_DATA_PATH, 
                               "facial_interventions_{}.json".format(video_name)), 'r') as facial_interventions_json_file, \
             open(os.path.join(config.VOCAL_INTERVENTIONS_PROCESSED_DATA_PATH, 
                               "vocal_interventions_{}.json".format(video_name)), 'r') as vocal_interventions_json_file:

            facial_interventions_content = json.load(facial_interventions_json_file)
            vocal_interventions_content = json.load(vocal_interventions_json_file)


        # DATOS PARA PROCESO DE CONTEO
        facial_begin_content = facial_interventions_content["begin"]
        facial_intervener_content = facial_interventions_content["intervener"]
        vocal_begin_content = vocal_interventions_content["begin"]
        vocal_finish_content = vocal_interventions_content["finish"]
        vocal_intervener_content = vocal_interventions_content["intervener"]

        total_interventions = len(vocal_intervener_content)
        actual_vocal_intervener = vocal_intervener_content[0]
        actual_vocal_begin = vocal_begin_content[0]
        vocal_checked_interventions = 1


        diarization_file_content = [] # TUPLAS CON EL PRINCIPIO Y FINAL DE INTERVENCIONES VOCALES Y EL DIPUTADO
        print("")

        # LÓGICA DE COMPROBACIÓN DE DIPUTADOS FACIAL Y VOCAL POR INTERVALOS DE TIEMPO
        while vocal_checked_interventions < total_interventions:

                # Como se repiten mucho los diputados de forma seguida por los mini silencios, vamos a recorrer todas sus intervenciones seguidas
                if vocal_intervener_content[vocal_checked_interventions] != actual_vocal_intervener:
                                
                        actual_intervener_last_intervention = vocal_checked_interventions - 1 # EL ANTERIOR, es la última vez que aparece
                        actual_vocal_finish = vocal_finish_content[actual_intervener_last_intervention]
                        facial_begin_indices_included, facial_finish_indices_included = self._get_included_indices(
                                facial_begin_content, actual_vocal_begin, actual_vocal_finish)

                        facial_interveners_included = facial_intervener_content[facial_begin_indices_included:facial_finish_indices_included]

                        # Obtención de diccionario de tiempo por diputado
                        facial_begin_included = facial_begin_content[facial_begin_indices_included:facial_finish_indices_included+1]
                        deputy_time_dict = {}

                        print(len(facial_begin_included))
                        print(len(facial_interveners_included))

                        for i, deputy in enumerate(facial_interveners_included):
                                if deputy not in deputy_time_dict:
                                    if deputy != "Diputado desconocido":
                                        deputy_time_dict[deputy] = facial_begin_included[i+1] - facial_begin_included[i]
                                else:
                                    deputy_time_dict[deputy] += facial_begin_included[i+1] - facial_begin_included[i]
                        
                        if len(deputy_time_dict.keys()) > 0:
                                facial_intervener_mode = max(deputy_time_dict, key=deputy_time_dict.get)

                                # Si el interviniente facial y vocal son distintos, ponemos desconocido
                                if actual_vocal_intervener != facial_intervener_mode:
                                        diarization_file_content.append((actual_vocal_begin, actual_vocal_finish, "Diputado desconocido"))
                                else:
                                        diarization_file_content.append((actual_vocal_begin, actual_vocal_finish, self._get_associated_name(actual_vocal_intervener)))

                        actual_vocal_begin = vocal_begin_content[vocal_checked_interventions]
                        actual_vocal_intervener = vocal_intervener_content[vocal_checked_interventions]

                vocal_checked_interventions += 1


        # TRATAMOS LA ÚLTIMA INTERVENCIÓN
        actual_vocal_finish = vocal_finish_content[-1]
        facial_begin_indices_included, facial_finish_indices_included = self._get_included_indices(
                                facial_begin_content, actual_vocal_begin, actual_vocal_finish)

        facial_interveners_included = facial_intervener_content[facial_begin_indices_included:facial_finish_indices_included]
        facial_begin_included = facial_begin_content[facial_begin_indices_included:facial_finish_indices_included+1]
        print(facial_interveners_included)
        print(facial_begin_included)

        deputy_time_dict = {}
        for i, deputy in enumerate(facial_interveners_included):
                if deputy not in deputy_time_dict:
                        if deputy != "Diputado desconocido":
                                deputy_time_dict[deputy] = facial_begin_included[i+1] - facial_begin_included[i]
                else:
                        deputy_time_dict[deputy] += facial_begin_included[i+1] - facial_begin_included[i]

        if len(deputy_time_dict.keys()) > 0:
                facial_intervener_mode = max(deputy_time_dict, key=deputy_time_dict.get)

                if actual_vocal_intervener != facial_intervener_mode:
                        diarization_file_content.append((actual_vocal_begin, actual_vocal_finish, "Diputado desconocido"))
                else:
                        diarization_file_content.append((actual_vocal_begin, actual_vocal_finish, self._get_associated_name(actual_vocal_intervener)))

        return diarization_file_content