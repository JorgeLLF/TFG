
import os
from abc import ABC
import json
import cv2
import numpy as np


import sys
sys.path.append(os.getcwd() + "/Code")
import Code.config as config
from Code.data_collecting.embeddings.embedding_calculators import FaceRecEmbeddingCalculator, VocRecEmbeddingCalculator


class EmbeddingsSaverToJSON(ABC):

    def __init__(self, embeddings_calculator, data_folder_path, embeddings_file_path, file_name):
        self._embeddings_calculator = embeddings_calculator
        self._data_folder_path = data_folder_path
        self._embeddings_file_path = embeddings_file_path
        self._file_name = file_name


    def save_embeddings_to_file(self):
        file_content = {'id': [], "file": [], 'embedding': []}

        for id_folder in os.listdir(self._data_folder_path):
            id_folder_path = os.path.join(self._data_folder_path, id_folder)

            if os.path.isdir(id_folder_path):

                for file in os.listdir(id_folder_path):
                    file_path = os.path.join(id_folder_path, file)

                    # En caso de ser una imagen, la leemos con opencv. Si no, pasamos la ruta simplemente
                    if isinstance(self, FacialEmbeddingsSaverToJSON):
                        data = cv2.imread(file_path)
                    else:
                        data = file_path

                    embedding = self._embeddings_calculator.calc_embedding(data)
                    if isinstance(embedding, np.ndarray):
                        file_content['id'].append(id_folder)
                        file_content["file"].append(file)
                        file_content['embedding'].append(embedding)

        self._write_embeddings(file_content, self._file_name)


    def _write_embeddings(self, file_content, file_name):
        with open(os.path.join(self._embeddings_file_path, file_name + ".json"), 'w') as json_file:
            json.dump(file_content, json_file, cls=NPArraySerializer)


# Escritores de embeddings, lo único que cambia son sus atributos (calculador de embeddings, rutas de guardado)
class FacialEmbeddingsSaverToJSON(EmbeddingsSaverToJSON):

    def __init__(self, file_name="facial_embeddings"):
        super().__init__(FaceRecEmbeddingCalculator(),
                         data_folder_path=config.FACIAL_REC_PROCESSED_DATA_PATH + "/MiembrosPleno",
                         embeddings_file_path=config.FACIAL_REC_PROCESSED_DATA_PATH,
                         file_name=file_name
                         )


class VocalEmbeddingsSaverToJSON(EmbeddingsSaverToJSON):

    def __init__(self, file_name="vocal_embeddings"):
        super().__init__(VocRecEmbeddingCalculator(),
                         data_folder_path=config.VOCAL_REC_PROCESSED_DATA_PATH + "/MiembrosPleno",
                         embeddings_file_path=config.VOCAL_REC_PROCESSED_DATA_PATH,
                         file_name=file_name
                         )


class NPArraySerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            json.JSONEncoder.default(self, obj)
