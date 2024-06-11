
import shutil
import os
import sys
from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
import time
import torch
import ffmpeg


sys.path.append(os.getcwd() + "/Code")
import Code.config as config
from Code.utils.utils import transform_seconds_to_time


# INTERFAZ PARA DETECCION DE CAMBIOS DE PLANO
class ShotBoundaryDetector(ABC):

    def __init__(self):
        self._boundaries_thres = 10

    @abstractmethod
    def detect_shot_boundaries(self, video):
        pass

    def _delete_near_shots_boundaries(self, shots_boundaries):
        index_to_delete = []

        for i in range(len(shots_boundaries) - 1):
            if shots_boundaries[i+1] - shots_boundaries[i] < self._boundaries_thres:
                index_to_delete.append(i)

        shots_boundaries = np.delete(shots_boundaries, index_to_delete)
        return shots_boundaries



class ColorHistogramShotBoundaryDetector(ShotBoundaryDetector):

    def __init__(self, color_model="YUV"):
        super().__init__()
        self._components_ranges = {"Y": [0, 256], "U": [0, 256],
                                 "V": [0, 256], "Cr": [0, 256],
                                 "Cb": [0, 256]}  # Rangos de valores para las componentes YUV
        self._color_model = cv.COLOR_BGR2YUV if color_model == "YUV" else cv.COLOR_BGR2YCrCb
        self._divergence_threshold = 0.2  # Probar entre 0.1 y 0.5. 0.3 es el valor a partir del cual las transiciones suaves ya no importan tanto


    def _color_hist_from_frame(self, frame):
        hist_channel1 = cv.calcHist([frame], [0], None, [256], self._components_ranges.get("Y"))
        if self._color_model == cv.COLOR_BGR2YUV:
            hist_channel2 = cv.calcHist([frame], [1], None, [256], self._components_ranges.get("U"))
            hist_channel3 = cv.calcHist([frame], [2], None, [256], self._components_ranges.get("V"))
        else:
            hist_channel2 = cv.calcHist([frame], [1], None, [256], self._components_ranges.get("Cr"))
            hist_channel3 = cv.calcHist([frame], [2], None, [256], self._components_ranges.get("Cb"))
        return np.concatenate((hist_channel1, hist_channel2, hist_channel3))


    def _pdf_from_hist(self, hist):  # Función de densidad de probabilidad
        return hist / np.sum(hist)

    def _list_mean(self,list):
        return sum(list) / len(list)

    def _kl_divergence(self,pdf_hist1, pdf_hist2):
        return np.sum(pdf_hist1 * np.log(pdf_hist1 / pdf_hist2))


    def detect_shot_boundaries(self, video):

        video_cap = cv.VideoCapture(video)
        video_frames_num = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        video_fps = video_cap.get(cv.CAP_PROP_FPS)

        if video_cap.isOpened():
            _, frame1 = video_cap.read()
            frame1 = cv.cvtColor(frame1, self._color_model)

            shot_boundaries = [0]
            frames_count = 1

            while True:
                if frames_count == video_frames_num:
                    break
                else:
                    _, frame2 = video_cap.read()
                    frame2 = cv.cvtColor(frame2, self._color_model)

                    color_hist_frame1 = self._color_hist_from_frame(frame1)
                    color_hist_frame2 = self._color_hist_from_frame(frame2)
                    color_hist_frame1 += 1 # Sumamos 1 para evitar el problema de la división por 0 en la divergencia KL
                    color_hist_frame2 += 1

                    pdf_color_hist1 = self._pdf_from_hist(color_hist_frame1)
                    pdf_color_hist2 = self._pdf_from_hist(color_hist_frame2)

                    if self._kl_divergence(pdf_color_hist1, pdf_color_hist2) > self._divergence_threshold:
                        shot_boundaries.append(frames_count)

                    frame1 = frame2
                    frames_count += 1


        video_cap.release()
        cv.destroyAllWindows()

        shot_boundaries = self._delete_near_shots_boundaries(
            np.array(shot_boundaries))  # ELIMINAMOS SHOTS BOUNDARIES CERCANOS

        return shot_boundaries, video_fps



class TransNetV2ShotBoundaryDetector(ShotBoundaryDetector):

    def __init__(self):
        super().__init__()
        self._transnet_path = config.TRANSNET_PATH
        self._transnet_weights_path = config.TRANSNET_PATH + "/transnetv2-pytorch-weights.pth"
        self._max_cut_minutes_length = 3.5


    def _create_transnet_model(self):

        # Comprobamos la disponibilidad de la transnet
        try:
            transnet_path = os.path.abspath(self._transnet_path)
            sys.path.append(transnet_path)
            from transnetv2_pytorch import TransNetV2
        except Exception as e:
            print(e)
            exit(-1)

        model = TransNetV2()
        transnet_weights_path = os.path.abspath(self._transnet_weights_path)
        transnet_weights = torch.load(transnet_weights_path)
        model.load_state_dict(transnet_weights)
        model.eval()
        return model


    def _make_video_cuts(self, video):

        # AÑADIMOS FFMPEG AL PATH
        actual_env_path = os.environ.get("PATH", "")
        os.environ["PATH"] = actual_env_path + os.pathsep + config.FFMPEG_PATH

        # Creamos carpeta para el video
        video_name = video.split("/")[-1].split(".")[0]
        video_cuts_path = os.path.join(config.VIDEOCUTS_PROCESSED_DATA_PATH, video_name)
        if not os.path.exists(video_cuts_path):
            try:
                os.makedirs(video_cuts_path)
            except Exception as e:
                print("No se pudo crear un directorio para recortes.")
                exit(-1)

        # Obtenemos duracion total del video
        video_seconds_duration = self._get_video_duration(video)

        # Hacemos recortes
        files = 1
        max_cut_seconds_length = int(self._max_cut_minutes_length * 60)
        last_recorded_second = 0
        for i in range(0, video_seconds_duration - max_cut_seconds_length, max_cut_seconds_length):
            cut_video_name = "{}_{:05d}.mp4".format(video_name, files)
            ss = transform_seconds_to_time(i)
            to = transform_seconds_to_time(i + max_cut_seconds_length)
            print(video_cuts_path + "/" + cut_video_name)
            ffmpeg.input(video, ss=ss, to=to)\
                .output(os.path.join(video_cuts_path, cut_video_name)).overwrite_output().run()
            files += 1
            last_recorded_second = i + max_cut_seconds_length

        # En caso de que el divisor no sea 0, quedará una porción pequeña al final, hay que obtenerla también
        if video_seconds_duration % max_cut_seconds_length != 0:
            cut_video_name = "{}_{:05d}.mp4".format(video_name, files)
            ffmpeg.input(video,
                         ss=transform_seconds_to_time(last_recorded_second),
                         to=transform_seconds_to_time(video_seconds_duration)). \
                output(video_cuts_path + "/" + cut_video_name).overwrite_output().run()


    def _get_video_duration(self, video):
        cap = cv.VideoCapture(video)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv.CAP_PROP_FPS)
        seconds_duration = int(total_frames / fps)
        cap.release()
        return seconds_duration


    def _get_predictions(self, model, video_tensor):

        with torch.no_grad():
            # dimensiones: batch, frames, height, width, channels. En este caso, [1, n_frames, 27, 48, 3]
            if torch.cuda.is_available():
                model = model.to("cuda")
                _, all_frame_predictions = model(video_tensor.to("cuda"))
                all_frame_predictions = torch.sigmoid(all_frame_predictions["many_hot"]).cpu().numpy()
            else:
                _, all_frame_predictions = model(video_tensor)
                all_frame_predictions = torch.sigmoid(all_frame_predictions["many_hot"]).numpy()
        return all_frame_predictions
    

    def _delete_video_cuts_folder(self, video):
        video_name = video.split("/")[-1].split(".")[0]
        video_cuts_path = os.path.join(config.VIDEOCUTS_PROCESSED_DATA_PATH, video_name)
        shutil.rmtree(video_cuts_path)


    def detect_shot_boundaries(self, video):

        # Definición del modelo
        model = self._create_transnet_model()

        # Obtenemos FPS
        video_cap = cv.VideoCapture(video)
        video_fps = video_cap.get(cv.CAP_PROP_FPS)
        video_cap.release()

        # Hacemos los recortes del video
        self._make_video_cuts(video)


        # PROCEDIMIENTO PRINCIPAL
        video_name = video.split("/")[-1].split(".")[0]
        video_cuts_path = os.path.join(config.VIDEOCUTS_PROCESSED_DATA_PATH, video_name)
        cut_shot_boundaries = []

        for video_cut in sorted(os.listdir(video_cuts_path)):
            single_video_cut_path = os.path.join(video_cuts_path, video_cut)

            # Guardado del vídeo de prueba en un array para pasarlo al modelo
            video_cap = cv.VideoCapture(single_video_cut_path)
            video_frames_num = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))

            video_array = np.zeros((video_frames_num, 27, 48, 3),
                                   dtype=np.uint8)  # 27, 48 porque es la misma relacion de dimensiones que 720, 1280

            frames_count = 0
            if video_cap.isOpened():
                while True:
                    ret, frame = video_cap.read()
                    if not ret:
                        break

                    frame = cv.resize(frame, (48, 27))
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                    video_array[frames_count, :, :, :] = frame
                    frames_count += 1

            video_cap.release()
            cv.destroyAllWindows()


            # Puesta a prueba del modelo
            video_tensor = torch.from_numpy(video_array)
            video_tensor = torch.unsqueeze(video_tensor, 0)  # devuelve tensor de dimension 1 en la dimension insertada

            predict_begin = time.time()
            all_frame_predictions = self._get_predictions(model, video_tensor)
            predict_finish = time.time()
            print("Tiempo del modelo para predicción del video {}: {} s".format(single_video_cut_path, predict_finish - predict_begin))
            print("")

            # Resultados
            test_video_predictions = all_frame_predictions.reshape(video_frames_num)
            shot_boundaries = np.array(np.where(test_video_predictions > 0.5))[0] # [[shot_boundaries]][0] = [shot_boundaries]

            cut_shot_boundaries.append(shot_boundaries)


        # BORRAMOS LA CARPETA CON LOS RECORTES ASOCIADOS AL VIDEO
        self._delete_video_cuts_folder(video)


        # LÓGICA DE ACOPLAMIENTO DE LOS CAMBIOS DE PLANO ENTRE RECORTES
        first_detections = np.insert(cut_shot_boundaries[0], 0, 0).tolist() # Añadimos el primer plano
        cut_frames = 60*self._max_cut_minutes_length*video_fps

        for i, cut in enumerate(cut_shot_boundaries[1:]):
            cut_location = (i+1)*cut_frames # Hacemos un desplazamiento
            first_detections.extend([shot_boundary + cut_location for shot_boundary in cut]) # Añadimos con el desplazamiento

        final_shot_boundaries = self._delete_near_shots_boundaries(
                np.array(first_detections))  # ELIMINAMOS SHOTS BOUNDARIES CERCANOS

        return final_shot_boundaries, video_fps