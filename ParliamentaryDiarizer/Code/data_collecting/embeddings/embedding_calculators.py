
from abc import ABC, abstractmethod
import torch
import face_recognition
from pyannote.audio import Model, Inference
from pyannote.core import Segment


class EmbeddingCalculator(ABC):

    @abstractmethod
    def calc_embedding(self, data):
        pass


class FaceRecEmbeddingCalculator(EmbeddingCalculator):

    def calc_embedding(self, face_image):
        face_embedding = face_recognition.face_encodings(face_image)
        if len(face_embedding) > 0:
            return face_embedding[0]
        else:
            return 0


class VocRecEmbeddingCalculator(EmbeddingCalculator):

    def __init__(self):
        self._auth_token = "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"
        self._embedding_inference = Inference(Model.from_pretrained("pyannote/embedding", use_auth_token=self._auth_token).to(
                                                                        "cuda" if torch.cuda.is_available() else "cpu"), 
                                                                        window="whole")

    def calc_embedding(self, voice_audio, segment_begin=00.00, segment_finish=10.00):
        segment = Segment(segment_begin, segment_finish) # Para el JSON de embeddings se usarán los segments por defecto, pero para el reconocimiento se hace una extracción de una porción justo en medio del audio
        return self._embedding_inference.crop(voice_audio, segment)



