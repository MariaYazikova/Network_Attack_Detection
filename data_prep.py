import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

file_path = 'datasets/cleaned_improved_cicids2017.csv'
df = pd.read_csv(file_path)

print("Размер датасета:", df.shape)
print("Первые 5 строк:\n", df.head())

missing = df.isna().sum()
print("Кол-во столбцов с пропусками:\n", (missing>0).sum())
print("\nТипы столбцов:\n", df.dtypes.value_counts())

print("\nУникальные классы (метки атак):")
print(df['Label'].unique())
    
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

df_filtered = df[df['Label'].isin(target_labels)].copy()
print("\nРазмер после фильтрации:", df_filtered.shape)
print("Оставшиеся классы:", df_filtered['Label'].unique())

le = LabelEncoder()
df_filtered['Label'] = le.fit_transform(df_filtered['Label'])

print("\nКлассы после кодирования:")
for i, cls in enumerate(le.classes_):
    print(f"{i}: {cls}")

X = df_filtered.drop("Label", axis=1)
y = df_filtered['Label']

print("\nРаспределение классов:")
print(y.value_counts(normalize=True))

const_cols = [col for col in X.columns if X[col].nunique() == 1]
print("\nКол-во константных признаков:", len(const_cols))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

print("\nРазмер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test, le), "datasets/prepared_data.pkl")
print("\nДанные сохранены в datasets/prepared_data.pkl")