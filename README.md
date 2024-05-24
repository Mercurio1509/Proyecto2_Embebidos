# Proyecto 2: Edge AI F1
*Taller de Sistemas Embebidos*

Descripción general de los archivos disponibles en este repositorio:

**carril.py:** Este archivo es un código de Python 3 que se utiliza para la detección de de las líneas de la calle.

**detect.tflite:** Modelo de aprendizaje autómatico convertido para Tensor-Flow Lite para la detección de objetos.

**core-image-base-raspberrypi4777.zip:** Este archivo comprimido tiene dentro la imagen de Yocto Project con todo lo necesario para correr los archivos de código, además de poderse conectar automáticamente a la red local predefinida.

**labelmap.txt:** Se utiliza para mapear las etiquetas numéricas de las clases a etiquetas legibles por humanos en modelos de clasificación. Este archivo es especialmente útil cuando se trabaja con modelos de clasificación, como los utilizados en problemas de reconocimiento de objetos o clasificación de imágenes.

**TFlite_detection_webcam.py:** Código final que lleva implementado la detección de objetos, detección de lineas de calle y activación de los motores atráves de los pines de la Raspberry Pi 4. Para poder ejecutar este código desde la terminal de la Raspberry Pi se debe hacer con el siguiente comando: `python3 TFLite_detection_webcam.py`.
