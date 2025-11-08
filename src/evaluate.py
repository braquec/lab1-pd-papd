import pandas as pd
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Cargar datos
df = pd.read_csv("data/processed_data.csv")
target_col = params['evaluate']['target']
X = df.drop(columns=[target_col])
y = df[target_col]

# Evaluar cada modelo
results = {}

for model_name in params['train']['models']:
    model_path = f"models/{model_name}.joblib"
    
    if not os.path.exists(model_path):
        print(f"Modelo no encontrado: {model_path}")
        continue
        
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    
    metrics = {}
    if "mse" in params['evaluate']['metrics']:
        metrics["mse"] = mean_squared_error(y, y_pred)
    if "r2" in params['evaluate']['metrics']:
        metrics["r2"] = r2_score(y, y_pred)
    if "mae" in params['evaluate']['metrics']:
        metrics["mae"] = mean_absolute_error(y, y_pred)
    
    results[model_name] = metrics
    print(f"Modelo {model_name}: {metrics}")

# Encontrar el mejor modelo basado en R² (mayor es mejor)
best_model = max(results.items(), key=lambda x: x[1].get('r2', -float('inf')))
results["best_model"] = {
    "name": best_model[0],
    "r2": best_model[1]["r2"],
    "mse": best_model[1]["mse"]
}

# Guardar métricas
with open("metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"\nMejor modelo: {best_model[0]} con R² = {best_model[1]['r2']:.4f}")