
import os

# ORIGINAL
# import Code.config as config
# from Code.audio_to_text_transcription.audio_transcriptors import WhisperAudioTranscriptor

# PARA VSCODE
import sys
import os
sys.path.append(os.getcwd() + "/Code")
import config
from audio_to_text_transcription.audio_transcriptors import WhisperAudioTranscriptor



# El PATH necesita tener la ruta al ffmpeg, por alguna razón
actual_env_path = os.environ.get("PATH", "")
os.environ["PATH"] = actual_env_path + os.pathsep + config.FFMPEG_PATH


# audio = config.PROCESSED_DATA_PATH + "/vocal_recognition/VideoAudios/Interventions/231025_01 - Trim_01/" \
#                                      "intervention_0001_231025_01 - Trim_01.wav"
audio = config.PROCESSED_DATA_PATH + "/vocal_recognition/VideoAudios/Interventions/231025_01 - Trim_01/" \
                                     "intervention_0001_231025_01 - Trim_01.wav"
# print(audio)
transcriptor = WhisperAudioTranscriptor()
print(transcriptor.transcript_audio_to_text(audio))