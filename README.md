# App para generación y clasificacion de imagenes

Esta aplicacion tiene 3 funcionalidades.En primer lugar permite generar imagenes segun el texto descrito.
Como segunda funcionalidad, permite clasificar un archivo de imagen.Es decir, la app te dirá que es lo que sale en la imagen que subes.
Como tercera funcionalidad la APP permite usar la imagen generada con la primera funcionalidad y clasificarla usando el modelo de clasificacion del punto 2.

Precondiciones:
1) Se debe contar con pytorch con CUDA habilitado. Para saber si cuda está habilitado en torch se puede usar:

import torch

torch.cuda.is_available()
