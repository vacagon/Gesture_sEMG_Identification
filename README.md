En este repositorio se encuentran los documentos adicionales del trabajo de fin de grado "Clasificación de señales de electromiografía
para la identificación de gestos de la mano".

En la carpeta "Archivos Simulink" se encuentran los modelos de Simulink que componen el sistema de identificación de gestos.

En la carpeta "Código Matlab" se encuentran 3 carpetas y 1 archivo llamado "TRAIN_MODELS.m". Es mediante este último que se entrenan los modelos, se validan, y se reentrenan. En la
carpeta "Datos señales" se encuentran las señales que se utilizan para entrenar y validar los modelos. En la carpeta de "Otras funciones" se encuentran las funciones para generar un
conjunto de datos, tanto de entrenamiento como de validación únicamente, a partir de las señales adquiridas, además de script con las constantes que carga "TRAIN_MODELS.m", y la fun-
ción "featyre_offline" que obtiene las características de manera offline haciendo uso de las funciones "MAV" "WL" "ZC" y "SSC" que calculan cada una de ellas. En la carpeta "Resulta-
dos" se encuentran los gráficos, modelos, precisiones, etc que se recogen en el capítulo de Resultados.

En la carpeta "Proyecto MS" se encuentra el documento de Microsft Project con la planificación del proyecto.
