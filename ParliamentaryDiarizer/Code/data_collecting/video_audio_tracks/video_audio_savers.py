
import ffmpeg
import cv2


import sys
import os
sys.path.append(os.getcwd() + "/Code")
import Code.config as config
from Code.utils.utils import transform_frames_to_time


class VideoAudioExtractor():

    def __init__(self):
        self._ffmpeg_path = config.FFMPEG_PATH
        self._channels = 1
        self._audio_sampling_frequency = "16k"
        self._saving_directory_path = config.VOCAL_REC_PROCESSED_DATA_PATH
        self._full_audios_folder = os.path.join(self._saving_directory_path, "VideoAudios/FullAudios")


    def extract_audio_portion(self, video_path, output_audio_folder_name, audio_name="voice_audio_001.wav",
                              ss="00:00:00", to="00:00:10"):
        actual_env_path = os.environ.get("PATH", "")
        os.environ["PATH"] = actual_env_path + os.pathsep + self._ffmpeg_path # Añadimos FFMPEG al PATH

        if not os.path.exists(os.path.join(self._saving_directory_path, output_audio_folder_name)):
            print(os.path.join(self._saving_directory_path, output_audio_folder_name))
            os.makedirs(os.path.join(self._saving_directory_path, output_audio_folder_name))

        # Ejecución del FFMPEG
        (
            ffmpeg.input(os.path.abspath(video_path), ss=ss, to=to)
                .output(os.path.join(self._saving_directory_path, output_audio_folder_name) + "/" + audio_name,
                        ac=self._channels, ar=self._audio_sampling_frequency)
                .overwrite_output()
                .run()
        )


    def extract_full_audio(self, video_path):

        total_time = self._calc_total_time(video_path)
        audio_name = video_path.split("/")[-1].split(".")[0]
        VideoAudioExtractor().extract_audio_portion(video_path=video_path,
                                                    output_audio_folder_name=self._full_audios_folder,
                                                    audio_name=audio_name + ".wav",
                                                    ss="00:00:00", to=total_time)

    def _calc_total_time(self, video_path):
        cap = cv2.VideoCapture(os.path.abspath(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(total_frames)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(fps)
        cap.release()
        total_time = transform_frames_to_time(total_frames, fps)
        return total_time
