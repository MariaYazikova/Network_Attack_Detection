import joblib
import pandas as pd

df_filtered = pd.read_csv('datasets/cleaned_improved_cicids2017.csv')

target_labels = [
    'BENIGN',
    'DoS Slowloris',
    'DoS Slowhttptest',
    'DoS Hulk',
    'DoS GoldenEye',
    'Infiltration - Portscan',
    'Portscan',
    'DDoS'
]

df_filtered = df_filtered[df_filtered['Label'].isin(target_labels)].copy()

X = df_filtered.drop("Label", axis=1)

_, _, _, _, le = joblib.load('datasets/prepared_data.pkl')

with open('mapping/features.txt', 'w', encoding='utf-8') as f:
    for col in X.columns:
        f.write(col + '\n')
        
with open('mapping/classes.txt', 'w', encoding='utf-8') as f:
    for cls in le.classes_:
        f.write(cls + '\n')

print("Классы записаны в mapping/classes.txt")
print("Признаки записаны в mapping/features.txt")