import joblib

xgb_model = joblib.load("models/saved/xgb_model.pkl")
xgb_model.get_booster().save_model("models/saved/xgb_model.json")
print("Модель сохранена в models/saved/xgb_model.json")