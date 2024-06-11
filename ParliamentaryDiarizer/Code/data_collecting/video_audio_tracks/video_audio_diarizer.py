
import os.path
import torch
from pyannote.audio import Pipeline

import sys
import os
sys.path.append(os.getcwd() + "/Code")
import Code.config as config


class VoiceActivityDetector():

    def __init__(self):
        self._auth_token = config.PYANNOTE_AUTH_TOKEN
        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self._auth_token).to(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def diarize_voice_activity(self, audio):
        diarization = self._pipeline(os.path.abspath(audio))
        return [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]

    def write_vad_to_file(self, audio, output_file_path):
        diarization = self._pipeline(os.path.abspath(audio))
        with open(os.path.abspath(output_file_path), "w") as diarization_file:
            diarization.write_rttm(diarization_file)
