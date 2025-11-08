import pandas as pd
import yaml
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Cargar datos procesados
df = pd.read_csv("data/processed_data.csv")

# Separar features y target
target_col = params['train']['target']
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Entrenando con {X.shape[0]} muestras y {X.shape[1]} features")
print(f"Target: {target_col}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['train']['test_size'], random_state=42
)

# Crear directorio de modelos si no existe
os.makedirs("models", exist_ok=True)

# Entrenar cada modelo configurado
for model_name in params['train']['models']:
    print(f"\nEntrenando modelo: {model_name}")
    
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "RandomForest":
        model = RandomForestRegressor(**params['train']['hyperparams']['RandomForest'])
    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor(**params['train']['hyperparams']['GradientBoosting'])
    else:
        print(f"Modelo no reconocido: {model_name}")
        continue
    
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Guardar modelo
    model_path = f"models/{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"Modelo guardado en: {model_path}")