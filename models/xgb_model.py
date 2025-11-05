import joblib
from xgboost import XGBClassifier

data_path = "datasets/prepared_data.pkl"
X_train_scaled, X_test_scaled, y_train, y_test, le = joblib.load(data_path)
    
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=len(le.classes_),
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_scaled, y_train)

joblib.dump(xgb, "models/saved/xgb_model.pkl")

print("Модель сохранена в models/saved/xgb_model.pkl")