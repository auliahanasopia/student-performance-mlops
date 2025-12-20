import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(path_raw, path_output):
    df = pd.read_csv(path_raw)
    df.columns = df.columns.str.strip()

    if 'G3' not in df.columns:
        raise ValueError("Kolom 'G3' tidak ditemukan di dataset")

    df['target'] = (df['G3'] >= 10).astype(int)
    df.drop(columns=['G3'], inplace=True)

    categorical_cols = df.select_dtypes(include='object').columns
    numerical_cols = df.select_dtypes(exclude='object').drop(columns=['target']).columns

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    df.to_csv(path_output, index=False)

if __name__ == "__main__":
    preprocess_data(
        r"C:\SMSML_Aulia-Hana-Sophiah\student-mat.csv",
        r"C:\SMSML_Aulia-Hana-Sophiah\studentperformance_preprocessing.csv"
    )

 
