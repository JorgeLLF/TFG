
from embeddings.embedding_file_savers import VocalEmbeddingsSaverToJSON


'''Cálculo y guardado manual de embeddings vocales en base de datos de embeddings vocales

    file_name = Nombre de salida de la base de datos de embeddings vocales

'''
embeddings_saver = VocalEmbeddingsSaverToJSON(file_name="vocal_embeddings")
embeddings_saver.save_embeddings_to_file()