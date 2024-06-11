
import numpy as np
from embeddings.embedding_file_readers import FacialEmbeddingsJSONReader


# PUESTA A PRUEBA DE LA LECTURA DE BASE DE DATOS DE EMBEDDINGS FACIALES

facial_embeddings_file_content = FacialEmbeddingsJSONReader(file_name="facial_embeddings").read_embeddings_from_file()
# print(facial_embeddings_file_content)
print(len(facial_embeddings_file_content["embedding"][0]))
print(type(np.array(facial_embeddings_file_content["embedding"])))
print(np.array(facial_embeddings_file_content["embedding"][0]))
