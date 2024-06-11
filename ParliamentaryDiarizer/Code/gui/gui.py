
# INTERFAZ GRÁFICA
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
sys.path.append(os.getcwd() + "/Code")
import Code.config as config

# PROCESOS
import time
from datetime import timedelta
import json
import torch
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, ToPILImage


import sys
import os
sys.path.append(os.getcwd() + "/Code")
from Code.shot_boundary_detection.detectors import ColorHistogramShotBoundaryDetector, TransNetV2ShotBoundaryDetector
from Code.shot_classification.modeling.models import ShotClassificationModel
from Code.utils.utils import get_specific_frame, load_model_classes_from_file, transform_seconds_to_time, \
    transform_frames_to_seconds, get_video_seconds
from Code.intervener_recognition.intervener_recognisers import FacialIntervenerRecogniser, VocalIntervenerRecogniser
from Code.data_collecting.embeddings.embedding_file_savers import NPArraySerializer
from Code.data_collecting.embeddings.embedding_file_readers import FacialEmbeddingsJSONReader, VocalEmbeddingsJSONReader
from Code.data_collecting.video_audio_tracks.video_audio_diarizer import VoiceActivityDetector
from Code.data_collecting.video_audio_tracks.video_audio_savers import VideoAudioExtractor
from Code.diarization_data_preprocessing.diarization_preprocessor import DiarizationPreprocessor
from Code.audio_to_text_transcription.audio_transcriptors import WhisperAudioTranscriptor
from Code.text_summarization.text_summarizers import HuggingFaceTextSummarizer, MRMSpanishFineTunedTextSummarizer, \
    PegasusTextSummarizer, TranslatingHuggingFaceTextSummarizer, GPTTextSummarizer



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cambiosPlanoButton = QtWidgets.QPushButton(self.centralwidget)
        self.cambiosPlanoButton.setEnabled(False)
        self.cambiosPlanoButton.setGeometry(QtCore.QRect(30, 110, 191, 51))
        self.cambiosPlanoButton.setAutoDefault(False)
        self.cambiosPlanoButton.setDefault(False)
        self.cambiosPlanoButton.setObjectName("cambiosPlanoButton")
        self.clasifPlanosButton = QtWidgets.QPushButton(self.centralwidget)
        self.clasifPlanosButton.setEnabled(False)
        self.clasifPlanosButton.setGeometry(QtCore.QRect(30, 170, 191, 51))
        self.clasifPlanosButton.setObjectName("clasifPlanosButton")
        self.reconButton = QtWidgets.QPushButton(self.centralwidget)
        self.reconButton.setEnabled(False)
        self.reconButton.setGeometry(QtCore.QRect(30, 230, 191, 51))
        self.reconButton.setObjectName("reconButton")
        self.titleLabel = QtWidgets.QLabel(self.centralwidget)
        self.titleLabel.setGeometry(QtCore.QRect(20, 10, 751, 91))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(16)
        self.titleLabel.setFont(font)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.titleLabel.setObjectName("titleLabel")
        self.genFicheroResultButton = QtWidgets.QPushButton(self.centralwidget)
        self.genFicheroResultButton.setEnabled(False)
        self.genFicheroResultButton.setGeometry(QtCore.QRect(30, 350, 191, 51))
        self.genFicheroResultButton.setObjectName("genFicheroResultButton")
        self.preprocessDiarizButton = QtWidgets.QPushButton(self.centralwidget)
        self.preprocessDiarizButton.setEnabled(False)
        self.preprocessDiarizButton.setGeometry(QtCore.QRect(30, 290, 191, 51))
        self.preprocessDiarizButton.setObjectName("preprocessDiarizButton")
        self.labelEstadoProceso = QtWidgets.QLabel(self.centralwidget)
        font = self.labelEstadoProceso.font()
        font.setPointSize(16)
        self.labelEstadoProceso.setFont(font)
        self.labelEstadoProceso.setGeometry(QtCore.QRect(350, 210, 110, 32))
        self.labelEstadoProceso.setText("")
        self.labelEstadoProceso.setObjectName("labelEstadoProceso")
        self.videoSeleccionadoLabel = QtWidgets.QLabel(self.centralwidget)
        self.videoSeleccionadoLabel.setGeometry(QtCore.QRect(62, 86, 55, 16))
        self.videoSeleccionadoLabel.setText("")
        self.videoSeleccionadoLabel.setObjectName("videoSeleccionadoLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.menuArchivo = QtWidgets.QMenu(self.menubar)
        self.menuArchivo.setObjectName("menuArchivo")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSeleccionar = QtWidgets.QAction(MainWindow)
        self.actionSeleccionar.setObjectName("actionSeleccionar")
        self.actionVerResumenes = QtWidgets.QAction(MainWindow)
        self.actionVerResumenes.setObjectName("actionVerResumenes")
        self.menuArchivo.addAction(self.actionSeleccionar)
        self.menuArchivo.addSeparator()
        self.menuArchivo.addAction(self.actionVerResumenes)
        self.menuArchivo.addSeparator()
        self.menubar.addAction(self.menuArchivo.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # self.numero = 1
        self.actionSeleccionar.triggered.connect(self.select_videos)
        self.actionVerResumenes.triggered.connect(self.see_summaries)
        self.cambiosPlanoButton.clicked.connect(lambda: self.processButton(self.cambiosPlanoButton,
                                                                           self.clasifPlanosButton,
                                                                           self.processCambiosPlano))
        self.clasifPlanosButton.clicked.connect(lambda: self.processButton(self.clasifPlanosButton,
                                                                           self.reconButton,
                                                                           self.processClasifPlanos))
        self.reconButton.clicked.connect(lambda: self.processButton(self.reconButton,
                                                                    self.preprocessDiarizButton,
                                                                    self.processRecon))
        self.preprocessDiarizButton.clicked.connect(lambda: self.processButton(self.preprocessDiarizButton,
                                                                               self.genFicheroResultButton,
                                                                               self.processPreprocessDiariz))
        self.genFicheroResultButton.clicked.connect(lambda: self.processButton(self.genFicheroResultButton,
                                                                               None,
                                                                               self.processGenFicheroResult))


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.cambiosPlanoButton.setText(_translate("MainWindow", "Detectar cambios de plano"))
        self.clasifPlanosButton.setText(_translate("MainWindow", "Clasificar planos"))
        self.reconButton.setText(_translate("MainWindow", "Reconocer diputado"))
        self.titleLabel.setText(_translate("MainWindow", "Diarizador de sesiones en el Parlamento de Canarias"))
        self.genFicheroResultButton.setText(_translate("MainWindow", "Generar resultado"))
        self.preprocessDiarizButton.setText(_translate("MainWindow", "Preprocesar diarización"))
        self.menuArchivo.setTitle(_translate("MainWindow", "Archivo"))
        self.actionSeleccionar.setText(_translate("MainWindow", "Seleccionar Video"))
        self.actionVerResumenes.setText(_translate("MainWindow", "Ver Resumenes"))


    def select_videos(self):
        options = QtWidgets.QFileDialog.Options()
        videos_data_path = config.VIDEOS_RAW_DATA_PATH # Directorio de vídeos
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self.centralwidget,
                                                             "Seleccionar Video",
                                                             videos_data_path,
                                                             "Videos (*.mp4 *.avi *.mov);;Todos los archivos (*)",
                                                             options=options)
        if files:
            self.video_list = files
            self.videoSeleccionadoLabel.setText(f"¡Videos seleccionados!")
            self.labelEstadoProceso.setText(f"")
            self.labelEstadoProceso.adjustSize()
            self.videoSeleccionadoLabel.adjustSize()
            self.cambiosPlanoButton.setEnabled(True)
            self.clasifPlanosButton.setEnabled(False)
            self.reconButton.setEnabled(False)
            self.preprocessDiarizButton.setEnabled(False)
            self.genFicheroResultButton.setEnabled(False)


    def see_summaries(self):
        summaries_data_path = config.SUMMARIES_PROCESSED_DATA_PATH
        QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "Abrir Explorador", directory=summaries_data_path)



    def processButton(self, actual_button, siguiente_boton, process):
        actual_button.setEnabled(False)

        # Activamos el siguiente botón si existe
        if siguiente_boton:
            siguiente_boton.setEnabled(True)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor) # Cambiamos el cursor a un cursor de espera

        process() # Realizamos proceso

        QtWidgets.QApplication.restoreOverrideCursor() # Restauramos el cursor al estado normal

        self.labelEstadoProceso.setText(f"Completado \n{actual_button.text()}.")
        self.labelEstadoProceso.adjustSize()



    def processCambiosPlano(self):

        self.begin = time.time()

        # video_list = [config.RAW_DATA_PATH + "/Videos/Validacion/PruebaLarga.mp4",
        #               config.RAW_DATA_PATH + "/Videos/Validacion/Prueba02_230912_01.mp4",
        #               config.RAW_DATA_PATH + "/Videos/Validacion/Prueba03_230912_02.mp4",
        #               config.RAW_DATA_PATH + "/Videos/Validacion/Prueba04_230912_02.mp4",
        #               config.RAW_DATA_PATH + "/Videos/Validacion/Prueba05_231121_01.mp4",
        #               config.RAW_DATA_PATH + "/Videos/Validacion/Prueba06_231121_01.mp4",
        #               config.RAW_DATA_PATH + "/Videos/Validacion/Prueba07_231011_01.mp4",
        #               config.RAW_DATA_PATH + "/Videos/Validacion/Prueba08_231121_01.mp4",
        #               config.RAW_DATA_PATH + "/Videos/Validacion/Prueba09_231107_02.mp4"]

        video_list = [config.RAW_DATA_PATH + "/Videos/Validacion/PruebaLarga.mp4"]

        print("")
        torch.cuda.empty_cache()


        # # DETECTOR DE CAMBIOS DE PLANO POR HISTOGRAMA DE COLOR
        # print("Detección de cambios de plano por histograma de color")
        # print("-----------------------------------------------------")
        # print("")
        # self.shot_boundary_detector = ColorHistogramShotBoundaryDetector(self.video_list)
        # self.videos_shots_boundaries_detections = {}
        #
        # for video in self.video_list:
        #     self.videos_shots_boundaries_detections[video] = self.shot_boundary_detector.detect_shot_boundaries(video)
        #
        # for video in self.videos_shots_boundaries_detections.keys():
        #     print("{}".format(video))
        #     video_detections = self.videos_shots_boundaries_detections[video]
        #     video_shots_boundaries = video_detections[0]
        #     video_fps = video_detections[1]
        #     for shot_boundary in video_shots_boundaries:
        #         time_delta = timedelta(seconds=shot_boundary * (1 / video_fps))
        #         print("Tiempo en hh:mm:ss, {}. ".format(time_delta))
        #
        #         # # Persistencia para comprobación
        #         # if not os.path.exists(config.PROCESSED_DATA_PATH + "/shot_boundary_detections/ColHist"):
        #         #     os.makedirs(config.PROCESSED_DATA_PATH + "/shot_boundary_detections/ColHist")
        #         # with open(os.path.join(config.PROCESSED_DATA_PATH + "/shot_boundary_detections/ColHist",
        #         #                        "ColHist_{}.txt".format(video.split("/")[-1].split(".")[0])), "a") as file:
        #         #     file.write("Tiempo en hh:mm:ss, {}. \n".format(time_delta))
        #         # print(shot_boundary)
        #     print("")


        # DETECTOR DE CAMBIOS DE PLANO POR TRANSNETV2
        print("")
        print("Detección de cambios de plano por red neuronal TransNetV2")
        print("---------------------------------------------------------")
        print("")
        self.shot_boundary_detector = TransNetV2ShotBoundaryDetector()
        self.videos_shots_boundaries_detections = {}

        for video in self.video_list:
            self.videos_shots_boundaries_detections[video] = self.shot_boundary_detector.detect_shot_boundaries(video)

        for video in self.videos_shots_boundaries_detections.keys():
            print("{}".format(video))
            video_detections = self.videos_shots_boundaries_detections[video]
            video_shots_boundaries = video_detections[0]
            video_fps = video_detections[1]
            for shot_boundary in video_shots_boundaries:
                time_delta = timedelta(seconds=shot_boundary * (1 / video_fps))
                print("Tiempo en hh:mm:ss, ", time_delta)

                # # Persistencia para comprobación
                # if not os.path.exists(config.PROCESSED_DATA_PATH + "/shot_boundary_detections/TransNet"):
                #     os.makedirs(config.PROCESSED_DATA_PATH + "/shot_boundary_detections/TransNet")
                # with open(os.path.join(config.PROCESSED_DATA_PATH + "/shot_boundary_detections/TransNet",
                #                        "TransNet_{}.txt".format(video.split("/")[-1].split(".")[0])), "a") as file:
                #     file.write("Tiempo en hh:mm:ss, {}. \n".format(time_delta))
            print("")



    def processClasifPlanos(self):

        # CLASIFICACIÓN DE LOS PLANOS
        print("")
        print("Clasificación de planos de sesiones parlamentarias")
        print("-------------------------------------------------")
        print("")

        shot_classification_classes = load_model_classes_from_file(config.SHOT_CLASSIFICATION_CODE_PATH +
                                                                   "/shot_classification_classes.json")

        device = ("cuda" if torch.cuda.is_available()
                  else "cpu")
        self.shot_classification_model = ShotClassificationModel()
        self.shot_classification_model = self.shot_classification_model.to(device)
        self.shot_classification_model.load_state_dict(torch.load(config.SHOT_CLASSIFICATION_CODE_PATH +
                                                             "/defDataset_WeightedShotClassificationModel.pth", # Cambiar si fuera necesario
                                                             map_location=torch.device("cpu")))
        self.shot_classification_model.eval()

        self.videos_shots_classifications = {}
        shot_transform = Compose([
            ToPILImage(),  # Paso que se hace de forma implícita en el modelo al crear los DataLoader
            Resize((227, 227)),
            ToImage(),
            ToDtype(torch.float32, scale=True)
        ])

        for video in self.videos_shots_boundaries_detections.keys():
            print("{}".format(video))
            video_detections = self.videos_shots_boundaries_detections[video]
            video_shots_boundaries = video_detections[0]
            video_shots_classifications = []

            shots_list = [get_specific_frame(video, shot_boundary + 10) for shot_boundary in
                          video_shots_boundaries]  # + 10 para evitar el problema de las transiciones suaves
            tensored_transformed_shots_list = torch.stack(
                [shot_transform(shot) for shot in shots_list])

            pred = self.shot_classification_model(tensored_transformed_shots_list.to(device))
            _, max_index_pred = torch.max(pred, 1)
            video_shots_classifications.extend(max_index_pred.tolist())

            for i, shot_boundary in enumerate(video_shots_boundaries):
                print("Plano en el frame {}: {}".format(shot_boundary, shot_classification_classes[str(video_shots_classifications[i])]))

                # # Persistencia para comprobación
                # if not os.path.exists(config.PROCESSED_DATA_PATH + "/shot_classifications"):
                #     os.makedirs(config.PROCESSED_DATA_PATH + "/shot_classifications")
                # with open(os.path.join(config.PROCESSED_DATA_PATH + "/shot_classifications",
                #                        "shot_classifications_{}.txt".format(video.split("/")[-1].split(".")[0])), "a") as file:
                #     file.write("Plano en el frame {}: {}\n".format(shot_boundary,
                #                                         shot_classification_classes[str(video_shots_classifications[i])]))
            print("")

            # Obtención de las clases asociadas
            self.videos_shots_classifications[video] = [shot_classification_classes[str(video_shots_classifications[i])]
                                                   for i in range(len(video_shots_boundaries))]

        print("")


    # RECONOCIMIENTO DE DIPUTADOS
    def processRecon(self):

        self.facial_intervener_recogniser = FacialIntervenerRecogniser()
        self.facial_embeddings_file_content = FacialEmbeddingsJSONReader().read_embeddings_from_file()
        self.video_audio_extractor = VideoAudioExtractor()
        self.vad = VoiceActivityDetector()
        self.vocal_intervener_recogniser = VocalIntervenerRecogniser()
        self.vocal_embeddings_file_content = VocalEmbeddingsJSONReader().read_embeddings_from_file()

        for video in self.videos_shots_boundaries_detections.keys():

            video_name = video.split("/")[-1].split(".")[0]

            # RECONOCIMIENTO FACIAL

            video_shots_boundaries = self.videos_shots_boundaries_detections[video][0]
            video_fps = self.videos_shots_boundaries_detections[video][
                1]  # Importante, para transformación de frames a segundos
            video_shots_types = self.videos_shots_classifications[video]

            json_pos_facial_interveners = []
            for i in range(1, len(video_shots_boundaries)):
                json_pos_facial_interveners.append(self.facial_intervener_recogniser.recognise(video,
                                                                                          video_shots_boundaries[i - 1],
                                                                                          video_shots_boundaries[i],
                                                                                          video_shots_types[i - 1]))
            print("")
            print(json_pos_facial_interveners)
            print("")

            facial_interveners_ids = []
            for intervener in json_pos_facial_interveners:
                if isinstance(intervener, str):
                    print(intervener)
                    facial_interveners_ids.append(intervener)  # Estamos pegando "Diputado desconocido"
                else:
                    print(
                        self.facial_embeddings_file_content["id"][intervener] + " " + self.facial_embeddings_file_content["file"][
                            intervener])
                    facial_interveners_ids.append(self.facial_embeddings_file_content["id"][intervener])
            print("")
            print("")

            # Directorio para guardado de intervenciones faciales
            if not os.path.exists(config.FACIAL_INTERVENTIONS_PROCESSED_DATA_PATH):
                os.makedirs(config.FACIAL_INTERVENTIONS_PROCESSED_DATA_PATH)

            seconds_transformed_frames = [transform_frames_to_seconds(frame, video_fps) for frame in
                                          video_shots_boundaries]
            seconds_transformed_frames.extend([get_video_seconds(video)])  # Importante, para cáclulo de tiempos posterior
            facial_interventions_file_content = {"begin": seconds_transformed_frames,
                                                 "type": video_shots_types,
                                                 "intervener": facial_interveners_ids}

            with open(os.path.join(config.FACIAL_INTERVENTIONS_PROCESSED_DATA_PATH,
                                   "facial_interventions_{}.json".format(video_name)),
                      'w') as facial_interventions_json_file:
                json.dump(facial_interventions_file_content, facial_interventions_json_file, cls=NPArraySerializer)


            # RECONOCIMIENTO VOCAL
            self.video_audio_extractor.extract_full_audio(video)
            audio = config.DATA_PATH + "/Processed/vocal_recognition/VideoAudios/FullAudios/" + video_name + ".wav"

            audio_diarization = self.vad.diarize_voice_activity(audio=audio)

            json_pos_vocal_interveners = []
            vocal_interveners_ids = []
            for intervention in audio_diarization:
                json_pos_vocal_interveners.append(self.vocal_intervener_recogniser.recognise(audio,
                                                                                        intervention[0],
                                                                                        intervention[1]))

            print("")
            print(json_pos_vocal_interveners)
            print("")

            for intervener in json_pos_vocal_interveners:
                if isinstance(intervener, int):
                    print("Intervención muy corta")
                else:
                    print(
                        self.vocal_embeddings_file_content["id"][intervener] + " " + self.vocal_embeddings_file_content["file"][
                            intervener])
                    vocal_interveners_ids.append(self.vocal_embeddings_file_content["id"][intervener])
            print("")
            print("")

            # Directorio para guardado de intervenciones faciales
            if not os.path.exists(config.VOCAL_INTERVENTIONS_PROCESSED_DATA_PATH):
                os.makedirs(config.VOCAL_INTERVENTIONS_PROCESSED_DATA_PATH)

            vocal_interventions_file_content = {"begin": [intervention[0] for intervention in audio_diarization
                                                          if intervention[1] - intervention[
                                                              0] > config.MIN_INTERVENTION_TIME_SECONDS],
                                                "finish": [intervention[1] for intervention in audio_diarization
                                                           if intervention[1] - intervention[
                                                               0] > config.MIN_INTERVENTION_TIME_SECONDS],
                                                "intervener": vocal_interveners_ids}

            with open(os.path.join(config.VOCAL_INTERVENTIONS_PROCESSED_DATA_PATH,
                                   "vocal_interventions_{}.json".format(video_name)),
                      'w') as vocal_interventions_json_file:
                json.dump(vocal_interventions_file_content, vocal_interventions_json_file, cls=NPArraySerializer)


        self.finish = time.time()
        print("Tiempo de ejecución total hasta el guardado de datos de las intervenciones: {} minutos".format(
            (self.finish - self.begin) / 60))



    # PREPROCESAMIENTO DE FICHERO DE DIARIZACIÓN
    def processPreprocessDiariz(self):

        self.diarization_preprocessor = DiarizationPreprocessor()
        self.preprocessed_diarizations = {}

        for video in self.video_list:
            video_name = video.split("/")[-1].split(".")[0]
            self.preprocessed_diarizations[video_name] = self.diarization_preprocessor.preprocess(video_name)

        print(self.preprocessed_diarizations)
        print("")



    # RESUMEN DE INTERVENCIONES
    def processGenFicheroResult(self):

        self.video_audio_extractor = VideoAudioExtractor()
        self.video_audio_transcriptor = WhisperAudioTranscriptor()
        # text_summarizer = HuggingFaceTextSummarizer()
        # text_summarizer = MRMSpanishFineTunedTextSummarizer()
        # text_summarizer = TranslatingHuggingFaceTextSummarizer()
        # text_summarizer = PegasusTextSummarizer()
        self.text_summarizer = GPTTextSummarizer()

        match self.text_summarizer:
            case _ if isinstance(self.text_summarizer, GPTTextSummarizer):
                model_name = "GPT"
            case _ if isinstance(self.text_summarizer, PegasusTextSummarizer):
                model_name = "PEG"
            case _ if isinstance(self.text_summarizer, HuggingFaceTextSummarizer):
                model_name = "HF_FB_BART"
            case _ if isinstance(self.text_summarizer, MRMSpanishFineTunedTextSummarizer):
                model_name = "MRM_SP_FineTuned"
            case _ if isinstance(self.text_summarizer, TranslatingHuggingFaceTextSummarizer):
                model_name = "HF_Translating"

        for video in self.video_list:
            video_name = video.split("/")[-1].split(".")[0]
            video_diarization = self.preprocessed_diarizations[video_name]
            audio_path = config.FULL_AUDIOS_VOCAL_REC_PROCESSED_DATA_PATH + "/" + video_name + ".wav"

            if not os.path.exists(config.SUMMARIES_PROCESSED_DATA_PATH):
                os.makedirs(config.SUMMARIES_PROCESSED_DATA_PATH)

            for i, intervention in enumerate(video_diarization):
                output_audio_folder_name = config.INTERVENTIONS_AUDIOS_VOCAL_REC_PROCESSED_DATA_PATH + "/" + video_name
                audio_name = "{}_intervention_{:04d}.wav".format(video_name, i + 1)
                ss = transform_seconds_to_time(intervention[0])
                to = transform_seconds_to_time(intervention[1])

                self.video_audio_extractor.extract_audio_portion(audio_path, output_audio_folder_name, audio_name,
                                                            ss, to)

                intervention_audio_path = output_audio_folder_name + "/" + audio_name
                transcribed_full_intervention = self.video_audio_transcriptor.transcript_audio_to_text(intervention_audio_path)

                # RESUMEN
                summarized_intervention = self.text_summarizer.summarize_text(transcribed_full_intervention)
                print("")
                print("[{} - {}] {}".format(ss, to, intervention[2]))
                print(summarized_intervention)
                print("")

                # Formateo de la string de resumen
                newline_positions = 100
                formatted_summarized_intervention = '\n'.join(summarized_intervention[i:i + newline_positions]
                                                              for i in
                                                              range(0, len(summarized_intervention), newline_positions))

                # Escritura a fichero
                with open(os.path.join(config.SUMMARIES_PROCESSED_DATA_PATH,
                                       "summary_{}_{}.txt".format(video_name, model_name)), "a") as summary_file:
                    summary_file.write("[{} - {}] {}\n".format(ss, to, intervention[2]))
                    summary_file.write("{}\n\n".format(formatted_summarized_intervention))