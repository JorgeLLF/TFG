
from video_audio_tracks.video_audio_savers import VideoAudioExtractor


'''Para obtención de audios manual para crear base de datos de embeddings vocales
    
    video_path = Video de entrada
    output_audio_folder_name = Carpeta de guardado en Processed/vocal_recognition
    voice_audio_name = Nombre del vídeo de salida
    ss = Comienzo de la porción de audio en el video
    to = Final de la porción de audio en el video

'''

video_path = 'E:/_TFG/Datos/Vídeos/231011_01.mp4'
output_audio_folder_name = "MiembrosPleno/Otros"
voice_audio_name = "voice_audio_0001.wav"
ss = "01:53:19"
to = "01:53:29"
VideoAudioExtractor().extract_audio_portion(video_path=video_path, output_audio_folder_name=output_audio_folder_name,
                                            ss=ss, to=to)

# VideoAudioExtractor().extract_full_audio(video_path)
