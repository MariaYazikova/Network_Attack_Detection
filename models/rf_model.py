import joblib
from sklearn.ensemble import RandomForestClassifier

data_path = "datasets/prepared_data.pkl"
X_train_scaled, X_test_scaled, y_train, y_test, le = joblib.load(data_path)
    
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

joblib.dump(rf, "models/saved/rf_model.pkl")

print("Модель сохранена в models/saved/rf_model.pkl")