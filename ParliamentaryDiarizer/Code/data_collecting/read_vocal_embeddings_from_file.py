
import numpy as np
from embeddings.embedding_file_readers import VocalEmbeddingsJSONReader


# PUESTA A PRUEBA DE LA LECTURA DE BASE DE DATOS DE EMBEDDINGS VOCALES

vocal_embeddings_file_content = VocalEmbeddingsJSONReader(file_name="vocal_embeddings").read_embeddings_from_file()
# print(vocal_embeddings_file_content)
print(len(vocal_embeddings_file_content["embedding"][0]))
print(type(np.array(vocal_embeddings_file_content["embedding"])))
print(np.array(vocal_embeddings_file_content["embedding"][0]))