# Usamos una imagen base oficial de Python
FROM python:3.9-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos e instalar requirements.txt
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Informamos a Docker que el contenedor está escuchando en el puerto 3000
EXPOSE 3000

# Comando para ejecutar la aplicación Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=3000"]