
from embeddings.embedding_file_savers import FacialEmbeddingsSaverToJSON


'''Cálculo y guardado manual de embeddings faciales en base de datos de embeddings faciales

    file_name = Nombre de salida de la base de datos de embeddings faciales
    
'''

embeddings_saver = FacialEmbeddingsSaverToJSON(file_name="facial_embeddings")
embeddings_saver.save_embeddings_to_file()