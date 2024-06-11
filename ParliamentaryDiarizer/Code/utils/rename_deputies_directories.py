
import os

def rename_deputies_directories(directory):
    # Obtenemos la lista de archivos en el directorio
    directories_list = os.listdir(directory)

    # Filtramos solo los archivos, no directorios
    directories = [deputy_directory for deputy_directory in directories_list
                   if os.path.isdir(os.path.join(directory, deputy_directory))]

    # Ordenamos los archivos alfabéticamente
    directories.sort()

    for i, deputy_directory in enumerate(directories):
        new_deputy_directory_name = deputy_directory.split(",")[-1].split()[-1]  # Asegura que siempre haya cuatro dígitos en el número
        old_path = os.path.join(directory, deputy_directory)
        new_path = os.path.join(directory, new_deputy_directory_name)

        # Renombramos el archivo
        os.rename(old_path, new_path)

        print(f"Renombrado: {deputy_directory} -> {new_deputy_directory_name}")


directory = "C:/Users/PC/Desktop/ULPGC/GCID/4º/TFG/Proyecto/ParliamentaryDiarizer/Data/Raw/MiembrosPleno"  # Reemplaza con la ruta de tu directorio
rename_deputies_directories(directory)