import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# Load dataset
try:
    data = pd.read_csv(r"C:\Users\Nagesh\Desktop\yash\projectmovie\data.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'data.csv' file not found.")
    exit()

# Data Cleaning
required_columns = ['director_name', 'duration', 'actor_1_name', 'budget', 'genres', 'title_year', 'gross']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Error: Missing columns in data.csv - {missing_columns}")
    exit()

# Drop missing values
data = data.dropna(subset=required_columns)
print(f"Data shape after removing rows with missing values: {data.shape}")

# Define features and target
X = data[['director_name', 'duration', 'actor_1_name', 'budget', 'genres', 'title_year']]
y = data['gross']

# Split into training and test sets (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numerical_features = ['duration', 'budget', 'title_year']
categorical_features = ['director_name', 'actor_1_name', 'genres']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define models
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'KNeighbors': KNeighborsRegressor()
}

# Training and selection
best_model = None
best_score = np.inf  # We're minimizing MAE
best_model_name = ""

for model_name, model in models.items():
    try:
        print(f"Evaluating {model_name}...")
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        avg_mae = -np.mean(scores)
        print(f'{model_name} Mean Absolute Error: {avg_mae:.2f}')
        if avg_mae < best_score:
            best_score = avg_mae
            best_model = pipe
            best_model_name = model_name
    except Exception as e:
        print(f"Error with model {model_name}: {e}")

# Fit best model and save it
if best_model:
    best_model.fit(X_train, y_train)
    with open('model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print(f"Training complete. Best model: {best_model_name}")
    print("Model saved as model.pkl.")

    # Evaluate on test set
    test_mae = np.mean(np.abs(best_model.predict(X_test) - y_test))
    print(f"Test Set MAE: {test_mae:.2f}")

    # Feature importance for RandomForest
    if best_model_name == 'RandomForest':
        try:
            feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
            importances = best_model.named_steps['model'].feature_importances_
            feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            top_features = feature_df.sort_values(by='Importance', ascending=False).head(10)
            print("\nTop 10 Important Features:")
            print(top_features)
        except Exception as e:
            print(f"Could not compute feature importance: {e}")
else:
    print("Error: No model was trained successfully.")
