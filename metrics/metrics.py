import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test, le = joblib.load("datasets/prepared_data.pkl")
models = {
    "LogisticRegression": joblib.load("models/saved/logreg_model.pkl"),
    "RandomForest": joblib.load("models/saved/rf_model.pkl"),
    "XGBoost": joblib.load("models/saved/xgb_model.pkl"),
}

results = []

with open("metrics/results.txt", "w", encoding="utf-8") as f:

    for name, model in models.items():
        f.write(f"\nМодель: {name}\n")

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
        f.write(f"Precision (weighted): {prec:.4f}\n")
        f.write(f"Recall (weighted): {rec:.4f}\n")
        f.write(f"F1 (macro): {f1_macro:.4f}\n")
        f.write(f"F1 (weighted): {f1_weighted:.4f}\n")

        f.write("\nКлассификационный отчёт:")
        f.write(classification_report(y_test, y_pred, target_names=le.classes_))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel('Предсказано')
        plt.ylabel('Истинное значение')
        plt.title(f'Матрица ошибок — {name}')
        plt.tight_layout()
        
        plot_path = f"metrics/plots/confusion_matrix_{name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        f.write(f"\nМатрица ошибок сохранена в {plot_path}\n")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "Precision_weighted": prec,
            "Recall_weighted": rec,
            "F1_macro": f1_macro,
            "F1_weighted": f1_weighted
        })

    df_results = pd.DataFrame(results)
    f.write("\nСводная таблица:")
    f.write(df_results.sort_values(by="Balanced_Accuracy", ascending=False).round(4).to_string(index=False))
    f.write("\n")

print("Результаты сохранены в metrics/results.txt")
