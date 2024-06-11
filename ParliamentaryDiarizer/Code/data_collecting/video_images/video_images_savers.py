
import cv2
import os
from abc import ABC, abstractmethod


import sys
sys.path.append(os.getcwd() + "/Code")
import Code.config as config
from Code.intervener_recognition.faces_locators import FaceRecFaceLocator


class VideoImagesSaver(ABC):

    def __init__(self, skipping_frames=15):
        self._skipping_frames = skipping_frames
        self._saving_directory_path = None


    def _get_last_frame_num_from_directory(self, directory):
        files = os.listdir(directory)
        frames_exist = [file for file in files if file.startswith('image_') and file.endswith('.png')]

        if not frames_exist:
            return 0

        frames_nums = [int(frame.split('_')[1].split('.')[0]) for frame in frames_exist]
        last_frame_num = max(frames_nums)
        return last_frame_num

    @abstractmethod
    def _get_images(self, frame):
        pass

    def save_video_images(self, video_path, label, begin_second, finish_second):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error al abrir el video.")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)

        begin_frame = int(begin_second * video_fps)
        finish_frame = int(finish_second * video_fps)


        label_path = os.path.join(self._saving_directory_path, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)

        # Obtenemos el número del último frame guardado en la carpeta
        last_frame_num = self._get_last_frame_num_from_directory(label_path)


        # Establecemos el punto de inicio en el frame correspondiente al segundo de inicio
        cap.set(cv2.CAP_PROP_POS_FRAMES, begin_frame)

        counter = 0
        for i in range(begin_frame, finish_frame + 1, self._skipping_frames):
            ret, frame = cap.read()

            if not ret:
                print(f"No se pudieron capturar más frames. Se alcanzó el final del video.")
                break

            images = self._get_images(frame)
            if not isinstance(images, list):
                continue

            for image in images:
                print(image)
                # Obtenemos el próximo número de frame disponible
                following_frame = last_frame_num + counter + 1
                counter += 1

                # Guardamos el frame como imagen en la carpeta destino
                image_file_name = f"image_{following_frame:04d}.png"
                saving_path = os.path.join(label_path, image_file_name)
                print(saving_path)
                cv2.imwrite(r"{}".format(saving_path), image)

        # Liberamos el objeto de captura
        cap.release()


class VideoFramesSaver(VideoImagesSaver):

    def __init__(self, skipping_frames=15):
        super().__init__(skipping_frames)
        self._saving_directory_path = config.MODELING_DATA_PATH + '/shot_classification'

    def _get_images(self, frame):
        return [frame]


class FaceRecVideoFacesSaver(VideoImagesSaver):

    def __init__(self, skipping_frames=15):
        super().__init__(skipping_frames)
        self._saving_directory_path = config.FACIAL_REC_PROCESSED_DATA_PATH + '/MiembrosPleno'
        self._face_locator = FaceRecFaceLocator()

    def _get_images(self, frame):
        return self._face_locator.locate_faces(frame)
