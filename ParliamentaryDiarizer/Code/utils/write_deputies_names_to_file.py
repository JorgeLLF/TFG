
import os

def write_deputies_name_to_file(directory, output_file):

    # Obtenemos la lista de nombres de carpetas en el directorio
    directories_names = [directory_name for directory_name in os.listdir(directory)
                         if os.path.isdir(os.path.join(directory, directory_name))]

    # Escribimos los nombres de las carpetas en el archivo
    with open(output_file, 'w') as file:
        for directory_name in directories_names:
            id = directory_name.split(",")[-1].split()[-1]
            deputy_name = " ".join(directory_name.split(",")[-1].split()[:-1])
            deputy_surname = directory_name.split(",")[0]
            file.write(id + "," + deputy_name + " " + deputy_surname + '\n')


# Especificamos la ruta del directorio a analizar
# directory = "../../Data/Raw/MiembrosPleno"
directory = "../../Data/Processed/intervener_recognition/MiembrosPleno"
# Especificamos el nombre del archivo en el que  escribir los nombres de las carpetas
output_file = "../../Data/Processed/facial_recognition/MiembrosPleno/deputies_names_copy.txt"

write_deputies_name_to_file(directory, output_file)

print(f"Los nombres de las carpetas se han guardado en {output_file}.")
