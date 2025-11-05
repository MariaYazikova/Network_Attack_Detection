import joblib
import numpy as np

prepared_path = 'datasets/prepared_data.pkl'
features_path = 'mapping/features.txt'
classes_path = 'mapping/classes.txt'
out_csv = 'inputs/inputs.csv'

X_train_scaled, X_test_scaled, y_train, y_test, le = joblib.load(prepared_path)

with open(features_path, 'r', encoding='utf-8') as f:
    features = [line.strip() for line in f if line.strip()]

with open(classes_path, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f if line.strip()]
    
y_test_arr = np.array(y_test)
X_test_arr = np.array(X_test_scaled)

rows = []
labels_intended = []
for cls_idx, cls_name in enumerate(classes):
    idxs = np.where(y_test_arr == cls_idx)[0]
    i = idxs[0]
    row = X_test_arr[i]
    rows.append(row)
    labels_intended.append((cls_idx, cls_name, i))

with open(out_csv, 'w', encoding='utf-8') as f:
    f.write(','.join(features) + '\n')
    for row in rows:
        line = ','.join([repr(float(x)) for x in row])
        f.write(line + '\n')

print(f"Записано {len(rows)} строк в {out_csv}")
print("индекс класса, имя класса, индекс строки:")
for t in labels_intended:
    print(t)