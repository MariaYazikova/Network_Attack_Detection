import joblib
from sklearn.linear_model import LogisticRegression

data_path = "datasets/prepared_data.pkl"
X_train_scaled, X_test_scaled, y_train, y_test, le = joblib.load(data_path)
    
logreg = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs',
    n_jobs=-1
)
logreg.fit(X_train_scaled, y_train)

joblib.dump(logreg, "models/saved/logreg_model.pkl")

print("Модель сохранена в models/saved/logreg_model.pkl")