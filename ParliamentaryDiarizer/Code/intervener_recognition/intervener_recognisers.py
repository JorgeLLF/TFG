
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


import sys
import os
sys.path.append(os.getcwd() + "/Code")
import Code.config as config
from Code.data_collecting.embeddings.embedding_file_readers import FacialEmbeddingsJSONReader, VocalEmbeddingsJSONReader
from Code.data_collecting.embeddings.embedding_calculators import FaceRecEmbeddingCalculator, VocRecEmbeddingCalculator
from Code.data_collecting.video_audio_tracks.video_audio_savers import VideoAudioExtractor
from Code.data_collecting.video_audio_tracks.video_audio_diarizer import VoiceActivityDetector
from Code.intervener_recognition.faces_locators import FaceRecFaceLocator


class IntervenerRecogniser:

    def __init__(self):
        self._distance_thres = 0.5
        self._shot_boundary_margin = 10

    def _calc_frequency_dict(self, indices, distances):
        frequency_dict = {}
        for i, dist in zip(indices, distances):
            if dist < self._distance_thres:
                if i not in frequency_dict:
                    frequency_dict[i] = 1
                else:
                    frequency_dict[i] += 1
        return frequency_dict

    def _get_nn(self, indices, distances):
        frequency_dict = self._calc_frequency_dict(indices[0], distances[0])
        if len(frequency_dict.keys()) >= 1:  # Si hay vecinos que cumplan el requisito, devolvemos el más cercano
            nearest_neighbor = max(frequency_dict, key=lambda x: frequency_dict[x])
        else:  # Si no los hay, devolvemos también el más cercano
            nearest_neighbor = indices[0][np.argmin(distances[0])]
        return nearest_neighbor


class FacialIntervenerRecogniser(IntervenerRecogniser):

    def __init__(self):
        super().__init__()
        self._facial_embeddings_file_content = FacialEmbeddingsJSONReader().read_embeddings_from_file()
        self._facial_comparator = NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(
            np.array(self._facial_embeddings_file_content["embedding"]))
        self._face_locator = FaceRecFaceLocator()
        self._facial_embedding_calculator = FaceRecEmbeddingCalculator()


    def recognise(self, video, actual_shot_boundary, next_shot_boundary, shot_type):  # En este caso, el actual y el next serán los cambios de plano, detectados mediante histograma de color o lo que sea

        distances, indices = None, None

        if shot_type == "PlanoCercano":
            test_shot_boundary = actual_shot_boundary + self._shot_boundary_margin  # Para COMENZAR en una imagen FUERA DE TRANSICIÓN SUAVE

            cap = cv2.VideoCapture(os.path.abspath(video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, test_shot_boundary)

            while test_shot_boundary <= next_shot_boundary:  # Vamos recorriendo los frames entre cambios de plano por si no encontrásemos cara
                print("Intento PlanoCercano")
                _, image = cap.read()

                # IMPORTANTE, Para seguir intentando obtener la imagen
                while not isinstance(image, np.ndarray):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, test_shot_boundary)
                    image = cap.read()

                face = self._face_locator.locate_faces(image)
                if len(face) == 1:
                    distances, indices = self._calc_nns(face[0])  # Obtenemos los vecinos más cercanos según la cara
                    if distances is None:
                        test_shot_boundary += 1
                        continue
                    else:
                        break

                test_shot_boundary += 1

            cap.release()

        elif shot_type == "PlanoMediano":
            test_shot_boundary = actual_shot_boundary + self._shot_boundary_margin  # Para COMENZAR en una imagen FUERA DE TRANSICIÓN SUAVE

            cap = cv2.VideoCapture(os.path.abspath(video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, test_shot_boundary)

            while test_shot_boundary <= next_shot_boundary:  # Vamos recorriendo los frames entre cambios de plano por si no encontrásemos cara
                print("Intento PlanoMediano")
                ret, image = cap.read()

                # IMPORTANTE, Para seguir intentando obtener la imagen
                while not isinstance(image, np.ndarray):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, test_shot_boundary)
                    image = cap.read()

                upper_body_image = self._detect_upper_body(image)  # Detectamos previamente el cuerpo
                if isinstance(upper_body_image, int):
                    test_shot_boundary += 1
                    continue

                face = self._face_locator.locate_faces(upper_body_image)
                if len(face) == 1:
                    distances, indices = self._calc_nns(face[0])
                    if distances is None:
                        test_shot_boundary += 1  # Si no se pudo calcular el embedding de la cara, se sigue iterando
                        continue
                    else:
                        break  # Si se pudo encontrar, ya se sale del bucle

                test_shot_boundary += 1

            cap.release()


        else:
            return "Diputado desconocido"

        if distances is None:
            return "Diputado desconocido"
        else:
            nearest_neighbor = self._get_nn(indices, distances)
            return nearest_neighbor


    def _calc_nns(self, face):

        face = np.ascontiguousarray(face[:, :, ::-1])

        face_embedding = self._facial_embedding_calculator.calc_embedding(face)
        if isinstance(face_embedding, int):
            distances, indices = None, None
        else:
            distances, indices = self._facial_comparator.kneighbors(face_embedding.reshape(1, -1))
        return distances, indices


    def _detect_upper_body(self, image):
        rows, columns = image.shape[0], image.shape[1]
        return image[int(rows * 0.1): int(rows * 0.7), int(columns * 0.4): int(columns * 0.6)]



class VocalIntervenerRecogniser(IntervenerRecogniser):

    def __init__(self):
        super().__init__()
        self._vocal_embeddings_file_content = VocalEmbeddingsJSONReader().read_embeddings_from_file()
        self._vocal_comparator = NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(
            np.array(self._vocal_embeddings_file_content["embedding"]))
        self._video_audio_extractor = VideoAudioExtractor()
        self._vocal_embedding_calculator = VocRecEmbeddingCalculator()
        self._interventions_audios_store = os.path.join(config.VOCAL_REC_PROCESSED_DATA_PATH,
                                                        "VideoAudios/Interventions")
        self._str_format_interventions_audios_store = "VideoAudios/Interventions/{}"
        self._vad = VoiceActivityDetector()
        self._max_crop_length = config.MAX_CROP_TIME_SECONDS
        self._min_intervention_time = config.MIN_INTERVENTION_TIME_SECONDS
        self._intervention_boundary = round(self._max_crop_length / 2, 2)


    def recognise(self, audio, intervention_begin, intervention_finish):  # En este caso, el actual y el next serán obtenidos mediante pyannote con el speaker diarization

        intervention_boundaries = intervention_finish - intervention_begin
        if intervention_boundaries < self._min_intervention_time:  # Intervenciones de menos de un segundo dan error, las eliminamos
            return 0

        full_video_name = audio.split("/")[-1].split(".")[0]
        video_interventions_folder_path = os.path.join(self._interventions_audios_store, full_video_name)
        if not os.path.exists(video_interventions_folder_path):
            os.makedirs(video_interventions_folder_path)

        # Recorte centrado
        distances, indices = self._calc_nns(audio=audio,
                                            segment_begin=round(intervention_begin + (intervention_boundaries / 2) - self._intervention_boundary, 2) 
                                                                if intervention_boundaries >= self._max_crop_length
                                                                else intervention_begin,
                                            segment_finish=round(intervention_finish - (intervention_boundaries / 2) + self._intervention_boundary, 2)
                                                                if intervention_boundaries >= self._max_crop_length
                                                                else intervention_finish
                                            )
        nearest_neighbor = self._get_nn(indices, distances)
        return nearest_neighbor


    def _calc_nns(self, audio, segment_begin, segment_finish):
        vocal_embedding = self._vocal_embedding_calculator.calc_embedding(audio,
                                                                          segment_begin=segment_begin,
                                                                          segment_finish=segment_finish
                                                                          )
        distances, indices = self._vocal_comparator.kneighbors(vocal_embedding.reshape(1, -1))
        return distances, indices