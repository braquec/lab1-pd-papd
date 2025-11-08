import pandas as pd
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)


df = pd.read_csv("data/dataset_v1.csv")

print(f"Dataset original: {df.shape}")
print("Columnas:", df.columns.tolist())

# Limpieza básica
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print(f"Dataset después de limpieza: {df.shape}")

# Verificar que existe la columna target
target_col = params['preprocess']['target']
if target_col not in df.columns:
    raise ValueError(f"Columna target '{target_col}' no encontrada. Columnas disponibles: {df.columns.tolist()}")

# En este dataset no hay variables categóricas, solo numéricas
print("Tipos de datos:")
print(df.dtypes)

# Guardar datos procesados
df.to_csv("data/processed_data.csv", index=False)
print("Preprocesamiento completado. Datos guardados en data/processed_data.csv")