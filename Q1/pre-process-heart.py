import pandas as pd
import sys

def encode_sex(input_file, output_file=None):
    df = pd.read_csv(input_file)

    if 'Sex' not in df.columns:
        raise ValueError("Column 'Sex' not found in the dataset.")

    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})

    if 'ExerciseAngina' not in df.columns:
        raise ValueError("Column 'ExerciseAngina' not found in the dataset.")

    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})

    if 'ST_Slope' not in df.columns:
        raise ValueError("Column 'ST_Slope' not found in the dataset.")

    df['ST_Slope'] = df['ST_Slope'].map({'Down': 0, 'Flat': 1, 'Up': 2})

    if 'ChestPainType' not in df.columns:
        raise ValueError("Column 'ChestPainType' not found in the dataset.")

    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'], dtype=int)

    if output_file is None:
        output_file = input_file.replace('.csv', '_encoded.csv')

    df.to_csv(output_file, index=False)
    print(f"Saved encoded file to: {output_file}")
    return df


input_path = "/home/nbalke/ml4h_data/p2/data/heart.csv"
output_path = "pre-processed-HEART.csv"
encode_sex(input_path, output_path)
