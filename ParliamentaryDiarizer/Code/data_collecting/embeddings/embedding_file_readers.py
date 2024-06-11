
import os
from abc import ABC
import json


import sys
sys.path.append(os.getcwd() + "/Code")
import Code.config as config


class EmbeddingsJSONReader(ABC):

    def __init__(self, embeddings_file_path, file_name):
        self._embeddings_file_path = embeddings_file_path
        self._file_name = file_name

    def read_embeddings_from_file(self):
        with open(os.path.join(self._embeddings_file_path, self._file_name + ".json"), "r") as embeddings_file:
            file_content = json.load(embeddings_file)
            return file_content


class FacialEmbeddingsJSONReader(EmbeddingsJSONReader):

    def __init__(self, file_name="facial_embeddings"):
        super().__init__(embeddings_file_path=config.FACIAL_REC_PROCESSED_DATA_PATH,
                         file_name=file_name)


class VocalEmbeddingsJSONReader(EmbeddingsJSONReader):
    def __init__(self, file_name="vocal_embeddings"):
        super().__init__(embeddings_file_path=config.VOCAL_REC_PROCESSED_DATA_PATH,
                         file_name=file_name)