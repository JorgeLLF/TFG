import os


def rename_images_at_directory(directory):
    # Obtenemos la lista de archivos en el directorio
    files_list = os.listdir(directory)

    # Filtramos solo los archivos, no directorios
    files = [file for file in files_list if os.path.isfile(os.path.join(directory, file))]

    # Ordenamos los archivos alfabéticamente
    files.sort()

    for i, file in enumerate(files):
        new_image_name = f"image_{i + 1:04d}.png"  # Asegura que siempre haya cuatro dígitos en el número
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_image_name)

        # Renombramos el archivo
        os.rename(old_path, new_path)

        print(f"Renombrado: {file} -> {new_image_name}")


directorio = "C:/Users/PC/Desktop/ULPGC/GCID/4º/TFG/Proyecto/ParliamentaryDiarizer/Data/Processed/intervener_recognition/MiembrosPleno/"  # Reemplaza con la ruta de tu directorio
label = "028"
rename_images_at_directory(directorio + label)
