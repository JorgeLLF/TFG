
from abc import ABC, abstractmethod
import numpy as np
import face_recognition


class FaceLocator(ABC):

    def __init__(self, face_margin=10):
        self._face_margin = face_margin

    @abstractmethod
    def locate_faces(self, frame):
        pass


class FaceRecFaceLocator(FaceLocator):

    def __init__(self, face_margin=10):
        super().__init__(face_margin)

    def locate_faces(self, frame):
        face_boxes = face_recognition.face_locations(frame)
        cropped_faces = []
        for face_box in face_boxes:
            y1, x2, y2, x1 = face_box
            y1 = 0 if y1 < self._face_margin else y1 - self._face_margin
            x1 = 0 if x1 < self._face_margin else x1 - self._face_margin
            y2 = frame.shape[0] if y2 > frame.shape[0] - self._face_margin else y2 + self._face_margin
            x2 = frame.shape[1] if x2 > frame.shape[1] - self._face_margin else x2 + self._face_margin
            cropped_faces.append(np.array(frame)[y1:y2, x1:x2])
        return cropped_faces

