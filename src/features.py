###############################
## Modificacion de variables ##
###############################

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

def zero_to_nan(df, features):
    """
    Reemplaza los valores 0 por NaN en las columnas especificadas del DataFrame.

    Parámetros:
    df (DataFrame): El DataFrame en el que se realizará la modificación.
    features (list): Lista de nombres de columnas donde se reemplazarán los valores 0 por NaN.

    Retorna:
    DataFrame: El DataFrame modificado con los valores 0 reemplazados por NaN en las columnas especificadas.
    """

    ret = df.copy()

    for feat in features:

        ret[feat] = ret[feat].replace(0, np.nan)
    
    return ret

def add_missing_indicators(df, features):
    """
    Agrega columnas que indican la presencia de valores faltantes en columnas específicas del DataFrame.

    Parámetros:
    df (DataFrame): El DataFrame al que se agregarán las columnas indicadoras.
    features (list): Lista de nombres de columnas para las cuales se crearán las columnas indicadoras de valores faltantes.

    Retorna:
    DataFrame: El DataFrame modificado con las nuevas columnas indicadoras de valores faltantes.
    """

    ret = df.copy()

    for feat in features:

        missing_ind_col = feat + '_missing_ind'

        ret[missing_ind_col] = ret[feat].isnull().astype(int)

    return ret

def impute_with_global_median(train, test, features):
    """
    Imputa los valores faltantes en las columnas especificadas de los DataFrames de entrenamiento y prueba con la mediana global.

    Parámetros:
    train (DataFrame): El DataFrame de entrenamiento en el que se realizará la imputación.
    test (DataFrame): El DataFrame de prueba en el que se realizará la imputación.
    features (list): Lista de nombres de columnas donde se imputarán los valores faltantes con la mediana global.

    Retorna:
    tuple: Una tupla que contiene los DataFrames modificados de entrenamiento y prueba con los valores faltantes imputados con la 
    mediana global en las columnas especificadas.
    """

    train_ret = train.copy()
    test_ret  = test.copy()

    for feat in features:

        global_median = train_ret[feat].median()

        train_ret[feat] = train_ret[feat].fillna(global_median)
        test_ret[feat]  = test_ret[feat].fillna(global_median)

    return train_ret, test_ret

def impute_median_by_outcome(train, test, features, target="Outcome"):
    """
    Imputa los valores faltantes en las columnas especificadas de los DataFrames de entrenamiento y prueba con la mediana calculada 
    según el valor de una columna objetivo.

    Parámetros:
    train (DataFrame): El DataFrame de entrenamiento en el que se realizará la imputación.
    test (DataFrame): El DataFrame de prueba en el que se realizará la imputación.
    features (list): Lista de nombres de columnas donde se imputarán los valores faltantes con la mediana por grupo.
    target (str): Nombre de la columna objetivo que se utilizará para agrupar los datos al calcular la mediana.

    Retorna:
    tuple: Una tupla que contiene los DataFrames modificados de entrenamiento y prueba con los valores faltantes imputados con la 
    mediana por grupo en las columnas especificadas.
    """

    train_ret = train.copy()
    test_ret  = test.copy()

    for feat in features:
        
        global_median = train_ret[feat].median(skipna=True)

        class_medians = train_ret.groupby(target)[feat].median()

        def fill_group(s):

            m = class_medians.loc[s.name]

            if pd.isna(m):

                m = global_median

            return s.fillna(m)

        train_ret[feat] = train_ret.groupby(target)[feat].transform(fill_group)
        test_ret[feat]  = test_ret[feat].fillna(global_median)

    return train_ret, test_ret

def impute_knn(train, test, features, n_neighbors=5):
    """
    Imputa los valores faltantes en las columnas especificadas de los DataFrames de entrenamiento y prueba utilizando el algoritmo 
    KNN.

    Parámetros:
    train (DataFrame): El DataFrame de entrenamiento en el que se realizará la imputación.
    test (DataFrame): El DataFrame de prueba en el que se realizará la imputación.
    features (list): Lista de nombres de columnas donde se imputarán los valores faltantes utilizando KNN.
    n_neighbors (int): Número de vecinos a considerar para la imputación KNN.

    Retorna:
    tuple: Una tupla que contiene los DataFrames modificados de entrenamiento y prueba con los valores faltantes imputados utilizando 
    KNN en las columnas especificadas.
    """

    train_ret = train.copy()
    test_ret  = test.copy()

    train_idx = train_ret.index
    test_idx = test_ret.index

    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(train_ret[features])
    X_test_scaled  = scaler.transform(test_ret[features])

    imputer = KNNImputer(n_neighbors=n_neighbors)

    X_train_imp = imputer.fit_transform(X_train_scaled)
    X_test_imp  = imputer.transform(X_test_scaled)

    X_train_final = scaler.inverse_transform(X_train_imp)
    X_test_final = scaler.inverse_transform(X_test_imp)

    train_ret[features] = pd.DataFrame(X_train_final, index=train_idx, columns=features)
    test_ret[features]  = pd.DataFrame(X_test_final, index=test_idx, columns=features)

    return train_ret, test_ret

def fit_by_medical_info(df):
    """
    Categoriza las columnas especificadas de los DataFrames de entrenamiento y prueba en función de valores médicos típicos

    Parámetros:
    df (DataFrame): El DataFrame en que se realizará la categorización.
    features (list): Lista de nombres de columnas donde se categorizarán los valores según información médica.

    Retorna:
    tuple: Una tupla que contiene los DataFrames modificados de entrenamiento y prueba con las columnas categorizadas según información médica.
    """

    ret = df.copy()

    ret["Glucose_bin"] = pd.cut(ret["Glucose"], bins=[-np.inf, 100, 125, np.inf], labels=["normal", "prediabetes", "diabetes"])

    ret["BMI_bin"] = pd.cut(ret["BMI"], bins=[-np.inf, 24, 30, np.inf], labels=["normal", "overweight", "obese"])

    ret["Age_bin"] = pd.cut(ret["Age"], bins=[-np.inf, 30, 45, np.inf], labels=["young", "adult", "senior"])

    ret["Pregnancies_bin"] = pd.cut(ret["Pregnancies"], bins=[-np.inf, 0, 2, np.inf], labels=["none", "low", "high"])

    return ret

def fit_quartile_bins(train, features, q=4):
    """
    Crea columnas categóricas en el DataFrame basadas en los cuartiles de las columnas especificadas basado en el DataFrame de entrenamiento.

    Parámetros:
    train (DataFrame): El DataFrame en el que se crearán las columnas categóricas.
    features (list): Lista de nombres de columnas donde se crearán las columnas categóricas basadas en cuartiles.
    q (int): Número de cuartiles a utilizar para la categorización.

    Retorna:
    DataFrame: El DataFrame modificado con las nuevas columnas categóricas basadas en cuartiles.
    """

    bins = {}

    for feat in features:

        quantiles = train[feat].quantile(q=np.linspace(0, 1, q + 1)).values
        quantiles = np.unique(quantiles)

        # Verificar si hay suficientes cuantiles únicos para crear las etiquetas

        if len(quantiles) < 3:

            bins[feat] = [-np.inf, np.inf]
            continue

        quantiles[0] = -np.inf
        quantiles[-1] = np.inf

        bins[feat] = quantiles.tolist()

    return bins

def apply_quartile_bins(df, bins):
    """
    Aplica las columnas categóricas basadas en cuartiles al DataFrame según los bins proporcionados.

    Parámetros:
    df (DataFrame): El DataFrame al que se aplicarán las columnas categóricas.
    bins (dict): Diccionario que contiene los bins para cada columna a categorizar.

    Retorna:
    DataFrame: El DataFrame modificado con las nuevas columnas categóricas basadas en cuartiles.
    """

    ret = df.copy()

    for feat, cutpoints in bins.items():

        new_col = f"{feat}_qbin"

        if len(cutpoints) == 2:

            ret[new_col] = "all"
            continue

        n_bins = len(cutpoints) - 1
        labels = [f"q{i}" for i in range(1, n_bins + 1)]

        ret[new_col] = pd.cut(ret[feat], bins=cutpoints, labels=labels, include_lowest=True).astype("object")

    return ret

def compute_woe_table(train_df, feature_bin, target="Outcome", eps=1e-6):
    """
    Calcula la tabla WOE (Weight of Evidence) para una variable categórica.

    Parámetros:
    train_df (DataFrame): El DataFrame de entrenamiento que contiene la variable categórica y la variable objetivo.
    feature_bin (str): El nombre de la columna categórica para la cual se calculará la tabla WOE.
    target (str): El nombre de la columna objetivo binaria.
    eps (float): Un valor pequeño para evitar divisiones por cero.

    Retorna:
    tuple: Una tupla que contiene la tabla WOE (DataFrame), el mapa WOE (diccionario) y el valor total de IV (float).
    """

    tmp = train_df[[feature_bin, target]].copy()

    grouped = tmp.groupby(feature_bin)[target]

    bad   = grouped.sum()
    total = grouped.count()
    good  = total - bad

    total_good = good.sum()
    total_bad  = bad.sum()

    woe_table = pd.DataFrame({"Good": good, "Bad": bad, "Total": total})

    woe_table["Dist_Good"] = (woe_table["Good"] + eps) / (total_good + eps*len(woe_table))
    woe_table["Dist_Bad"]  = (woe_table["Bad"]  + eps) / (total_bad  + eps*len(woe_table))

    woe_table["WOE"] = np.log(woe_table["Dist_Good"] / woe_table["Dist_Bad"])

    woe_table["IV_component"] = (woe_table["Dist_Good"] - woe_table["Dist_Bad"]) * woe_table["WOE"]

    iv_total  = woe_table["IV_component"].sum()
    woe_table = woe_table.sort_index()
    woe_map   = woe_table["WOE"].to_dict()

    return woe_table, woe_map, iv_total


def apply_woe(df, feature_bin, woe_map, new_name=None, default_woe=0.0):
    """
    Aplica la transformación WOE (Weight of Evidence) a una variable categórica en un DataFrame.

    Parámetros:
    df (DataFrame): El DataFrame al que se aplicará la transformación WOE.
    feature_bin (str): El nombre de la columna categórica a transformar.
    woe_map (dict): Un diccionario que mapea las categorías a sus valores WOE correspondientes.
    new_name (str): El nombre de la nueva columna transformada.
    default_woe (float): El valor por defecto para categorías no presentes en el mapa WOE.

    Retorna:
    DataFrame: El DataFrame con la nueva columna transformada.
    """
    ret = df.copy()

    if new_name is None:

        new_name = feature_bin + "_woe"

    ret[new_name] = (ret[feature_bin].astype("object").map(woe_map).astype(float).fillna(float(default_woe)))

    return ret

def one_hot_train_test(train_df, test_df, cat_features, drop_first=False):
    """
    One-hot encoding consistente entre train y test.

    Parámetros:
    train_df (DataFrame): El DataFrame de entrenamiento.
    test_df (DataFrame): El DataFrame de prueba.
    cat_features (list): Lista de nombres de columnas categóricas a codificar.
    drop_first (bool): Si es True, se eliminará la primera categoría para evitar la multicolinealidad.
    """

    Xtr = pd.get_dummies(train_df, columns=cat_features, drop_first=drop_first)
    Xte = pd.get_dummies(test_df,  columns=cat_features, drop_first=drop_first)

    Xtr, Xte = Xtr.align(Xte, join="left", axis=1, fill_value=0)

    return Xtr, Xte