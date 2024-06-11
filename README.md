# TFG

###Repositorio de TFG "Diarización y resumen de intervenciones parlamentarias en el Parlamento de Canarias"

- Se debe introducir una API_KEY de pago de OpenAI, así como un token de autenticación de pyannote.audio, en el fichero 
config.py en el módulo Code, en las variables OPENAI_API_KEY y PYANNOTE_AUTH_TOKEN.


- En caso de tener una máquina con alta capacidad en GPU (mínimo 12 GB) e interfaz gráfica,
se puede usar la versión de interfaz gráfica, ejecutando el script main_with_gui.py 
En caso de no contar con la interfaz gráfica, usar la versión main.py


- En cualquier caso, la detección de cambios de plano usada es la detección por red neuronal.
En caso de querer usar la detección por histograma de color, descomentar el bloque de código correspondiente
en main.py o en main_with_gui.py (según la versión probada) y comentar el bloque de red neuronal. Lo mismo para probar
otros resumidores, comentar y descomentar en los scripts ya mencionados en función del que se quiera
probar.
