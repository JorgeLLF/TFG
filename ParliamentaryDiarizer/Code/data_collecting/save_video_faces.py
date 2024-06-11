
from video_images.video_images_savers import FaceRecVideoFacesSaver


'''
    Guardado manual de imágenes faciales encontradas en video de entrada

    video_path = Video de entrada
    begin_second = Segundo de comienzo a partir del cual hacer la recolección en el vídeo de entrada
    finish_second = Segundo de final hasta el cual hacer la recolección en el vídeo de entrada
    skipping_frames = Cada cuantos frames se quiere intentar hacer la búsqueda facial
    label = Directorio en Data/Processed/facial_recognition/MiembrosPleno donde se quieren guardar las imágenes faciales

'''

video_path = 'E:/_TFG/Datos/Vídeos/231010_02.mp4'
begin_second = 2*3600 + 51*60 + 54
finish_second = 2*3600 + 51*60 + 64

# GUARDAR IMÁGENES DE CARAS
video_images_saver = FaceRecVideoFacesSaver(skipping_frames=30)
label = "Otros"
video_images_saver.save_video_images(video_path, label, begin_second, finish_second)