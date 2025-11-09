import pandas as pd

df = pd.read_csv('data/dataset_v1.csv')

df_v2 = df.copy()
df_v2 = df_v2.dropna()  # Eliminar nulos
df_v2 = df_v2.drop_duplicates()  # Eliminar duplicados

df_v2.to_csv('data/dataset_v2.csv', index=False)
print(f"Dataset v2 creado: {df_v2.shape}")