# Caso de Uso: Venta Cruzada con SQL y Vetiver (MLOps)

Este repositorio muestra un **ejemplo práctico de venta cruzada** utilizando una **base de datos SQL** como fuente de datos y el paquete **`vetiver`** para aplicar buenas prácticas de **MLOps** en modelos de machine learning.

El objetivo es ilustrar un flujo completo: carga de datos, entrenamiento del modelo, versionado y preparación para despliegue.

## Descripción del Caso de Uso

Se parte de información de clientes almacenada en una base de datos SQL para:

- Analizar patrones de comportamiento
- Entrenar un modelo de *cross-selling*
- Gestionar el modelo mediante `vetiver`
- Facilitar su reproducibilidad y mantenimiento

## Requisitos

Utiliza un entorno de conda:

```
conda create -n mlops_database_project python=3.10
```

El proyecto utiliza las siguientes paquetes y versiones:

```
scikit-learn==1.4.0
pandas==2.2.1
PyYAML==6.0.1
pyarrow==15.0.0
fastparquet==2024.2.0
vetiver==0.2.6
sqlalchemy==2.0.43
jupyterlab==4.1.5
```
Instala dichos paquetes instalar utilizando la siguiente instrucción:

```
pip install -r requirements.txt
```
