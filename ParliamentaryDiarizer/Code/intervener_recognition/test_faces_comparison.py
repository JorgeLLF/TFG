
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from Code.data_collecting.embeddings.embedding_file_readers import FacialEmbeddingsJSONReader
from Code.data_collecting.embeddings.embedding_calculators import FaceRecEmbeddingCalculator


facial_embeddings_file_content = FacialEmbeddingsJSONReader().read_embeddings_from_file()
near_neighbors = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(np.array(facial_embeddings_file_content["embedding"]))

test_image_embedding = FaceRecEmbeddingCalculator().calc_embedding(cv2.imread("../../Data/Processed/facial_recognition/MiembrosPleno/009/image_0001.png"))
distances, indices = near_neighbors.kneighbors(test_image_embedding.reshape(1, -1))
print("Vecinos más cercanos: {} y {}".format(
    facial_embeddings_file_content["id"][indices[0][0]] + " " + facial_embeddings_file_content["image"][indices[0][0]],
    facial_embeddings_file_content["id"][indices[0][1]] + " " + facial_embeddings_file_content["image"][indices[0][1]]))
print(distances)