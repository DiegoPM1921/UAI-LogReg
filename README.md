# Despliegue de Modelo de ML en Cloud

Este proyecto busca implementar una API REST para la predicción de diabetes a partir de variables clínicas, utilizando un modelo de regresión logística. El set de datos corresponde a [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), el cual se encuentra disponible en Kaggle.

La API fue desarrollada con FastAPI y desplegada utilizando Google Cloud.

## Contenido

Se detalla a continuación cada una de las rutas del repositorio.

- "/app": Código modular en Python para ejecutar la API.
- "/data": Contiene el set de datos crudo
- "/scr": Código modular en Python. El archivo $\verb|feautures.py|$ contiene el tratamiento de variables mientras que el archivo $\verb|train.py|$ se encarga de entrenar el modelo. 
- "/models": Carpeta donde se almacena el modelo de regresión logística

## Análisis exploratorio

Durante el análisis exploratorio de los datos se identificaron los siguientes puntos relevantes:

- **Variable objetivo desbalanceada**: Existe una mayor proporción de personas sin diabetes en el conjunto de datos.
- **Datos faltantes implícitos**: A pesar de que el set de datos no contiene valores nulos explícitos, sí presenta ceros no fisiológicos que representan valores faltantes. Las variables afectadas son:
  1. `Glucose`
  2. `BloodPressure`
  3. `SkinThickness`
  4. `Insulin`
  5. `BMI`
- **Diferencias en las distribuciones**: Algunas variables muestran cambios significativos en sus distribuciones al separarlas según la variable objetivo.

## Tratamiento de valores faltantes

Dado que el propósito del proyecto es la clasificación binaria, los valores cero (no fisiológicos) fueron tratados y reemplazados con el fin de no perder información relevante. Para el tratamiento de valores faltantes, se hizo una **imputación mediante el algoritmo K-Nearest Neighbors (KNN)**

## Feature engineering

### Categorización de variables

Con el objetivo de mejorar la capacidad predictiva del modelo, se categorizó el conjunto de datos utilizando dos enfoques distintos:

1. **Categorización por valores clínicos**: Las variables `Glucose`, `BMI`, `Age` y `Pregnancies` se categorizaron utilizando rangos clínicos comúnmente aceptados.
2. **Categorización por cuartiles**: Las variables `BloodPressure`, `SkinThickness`, `Insulin` y `DiabetesPedigreeFunction` se dividieron en cuartiles según su distribución. Estos cuartiles fueron calculados únicamente sobre el conjunto de entrenamiento para evitar *data leakage*.
3. **Indicadores de valores faltantes**: Para cada variable imputada se agregó una variable indicadora que señala la presencia o ausencia de datos originales, permitiendo capturar patrones asociados a los valores faltantes.

### Codificación de variables

Posterior a la categorización, se codificaron las variables siguiendo la metodología **Weight of Evidence (WOE)**. 

## Modelado

Se entrenó un modelo para la predicción de diabetes. El desempeño del modelo se midió ocupando las siguientes métricas.

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Resultados y análisis

A continuación, se muestran las métricas obtenidas para el modelo entrenado.

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| KNN + WOE | 0.7403 | 0.6000 | 0.7778 | 0.6774 | 0.8370 |

En contextos clínicos, un buen modelo no es necesariamente aquel con las mejores métricas globales, sino aquel que minimiza los errores más costosos. En este caso, resulta más crítico evitar clasificar a un paciente con diabetes como no diabético que el error contrario. Por lo tanto, el análisis del Recall cobra especial relevancia frente a otras métricas. Maximizar este valor puede ser el resultado más valioso en estos contextos.