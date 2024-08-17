import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from category_encoders import TargetEncoder

# Load data
train_data = pd.read_csv(r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-train-encoded.csv')
test_data = pd.read_csv(r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-test-encoded.csv')

# Prepare features and labels
X = train_data.drop(columns=['id_num', 'political_affiliation'])
y = train_data['political_affiliation']

# Target encoding for categorical features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
target_encoder = TargetEncoder(cols=categorical_cols)
X_encoded = target_encoder.fit_transform(X, y)

# Apply the same encoding to the test data
X_test = test_data.drop(columns=['id_num'])
X_test_encoded = target_encoder.transform(X_test)

# Add polynomial and interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_encoded)
X_test_poly = poly.transform(X_test_encoded)

# Standardization
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)

# Feature selection
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='mean')
selector.fit(X_poly_scaled, y)
X_selected = selector.transform(X_poly_scaled)
X_test_selected = selector.transform(X_test_poly_scaled)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', 0.5]  # Use valid values instead of 'auto'
}

# Initialize the random forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Use GridSearchCV for hyperparameter search and save results for each combination
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, return_train_score=True)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(f'Best Parameters: {grid_search.best_params_}')

# Predict on the test set and save results for each parameter combination
for params, mean_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
    # Train the model using each parameter combination
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the validation set
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Params: {params} | Validation Accuracy: {accuracy:.4f}')
    
    # Predict on the test set
    test_predictions = model.predict(X_test_selected)
    
    # Convert the predictions to the original political affiliation labels
    label_to_affiliation = {
        0: 'Democrat',
        1: 'Independent',
        2: 'Republican'
    }
    predicted_affiliations = [label_to_affiliation[pred] for pred in test_predictions]
    
    # Prepare the DataFrame for submission
    final_predictions = pd.DataFrame({
        'id_num': test_data['id_num'],
        'political_affiliation_predicted': predicted_affiliations
    })
    
    # Construct the file name
    file_name = f"CAH-201803-predictions-rf-n{params['n_estimators']}-md{params['max_depth']}-ms{params['min_samples_split']}-ml{params['min_samples_leaf']}-mf{params['max_features']}.csv"
    
    # Save the final predictions
    final_predictions.to_csv(rf'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\{file_name}', index=False)
    print(f"Results saved to: {file_name}")

print("All results have been saved.")
