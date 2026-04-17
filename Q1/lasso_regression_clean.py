import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=10000, l1_ratio=1, solver="saga"))])
param_grid = {"scaler": [StandardScaler(), 'passthrough'], "model__C" :[0.1,1,10]}
grid = GridSearchCV(pipeline, param_grid=param_grid, cv = 5, n_jobs=1)


#Reading Data
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
data:pd.DataFrame = pd.read_csv("heart.csv")
data = encode_sex("heart.csv")
X = data.drop(columns = ["HeartDisease"])
labels = pd.read_csv("heart.csv")["HeartDisease"]

#optimizing
grid.fit(X, labels)
print(grid.best_params_)
model = grid.best_estimator_.named_steps["model"]
#visualizing weights
weights = model.coef_[0]
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'value' : weights})
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
# We plot the actual Weight (not absolute) so we can see positive/negative impact
plt.barh(importance_df['feature'], importance_df['value'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importances (Lasso Logistic Regression)')
plt.gca().invert_yaxis() # Puts the most important feature at the top
plt.show()

