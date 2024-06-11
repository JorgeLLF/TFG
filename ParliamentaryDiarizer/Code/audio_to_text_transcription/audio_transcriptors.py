
import os
from abc import ABC, abstractmethod
import torch
import whisper


class AudioTranscriptor(ABC):

    @abstractmethod
    def transcript_audio_to_text(self, audio):
        pass


class WhisperAudioTranscriptor(AudioTranscriptor):

    def __init__(self):
        self._model = whisper.load_model("base")

    def transcript_audio_to_text(self, audio):
        transcription = self._model.to("cuda" if torch.cuda.is_available() else "cpu").transcribe(os.path.abspath(audio))
        return transcription["text"]
