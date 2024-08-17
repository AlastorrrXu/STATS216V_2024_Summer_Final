import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Load data
train_data = pd.read_csv(r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-train-encoded.csv')
test_data = pd.read_csv(r'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\CAH-201803-test-encoded.csv')

# Feature engineering function
def feature_engineering(X):
    # Polynomial features and interaction features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    return X_scaled

# Neural network model creation function
def create_model(input_shape, learning_rate=0.001, dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))  # Assuming 3 classes: Democrat, Independent, Republican
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Mapping party affiliation information
affiliation_mapping = {0: 'Democrat', 1: 'Independent', 2: 'Republican'}

# Manual hyperparameter tuning and saving predictions for each model
def manual_grid_search(X, y, test_data, param_grid, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for batch_size in param_grid['batch_size']:
        for epochs in param_grid['epochs']:
            for learning_rate in param_grid['learning_rate']:
                for dropout_rate in param_grid['dropout_rate']:
                    fold_scores = []
                    for train_index, val_index in skf.split(X, y):
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]
                        
                        model = create_model(X_train.shape[1], learning_rate, dropout_rate)
                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                        
                        y_val_pred = np.argmax(model.predict(X_val), axis=1)
                        accuracy = accuracy_score(y_val, y_val_pred)
                        fold_scores.append(accuracy)
                    
                    mean_score = np.mean(fold_scores)
                    print(f'Params: batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}, dropout_rate={dropout_rate} | CV Accuracy: {mean_score:.4f}')
                    
                    # Predict on the test set
                    X_test = feature_engineering(test_data.drop(columns=['id_num']).values)
                    test_predictions = np.argmax(model.predict(X_test), axis=1)
                    
                    # Convert numeric predictions to corresponding party affiliation names
                    predicted_labels = [affiliation_mapping[pred] for pred in test_predictions]
                    
                    # Save the predictions
                    predictions_df = pd.DataFrame({
                        'id_num': test_data['id_num'],
                        'political_affiliation_predicted': predicted_labels
                    })
                    
                    file_name = f'CAH-201803-predictions-nn-bs{batch_size}-ep{epochs}-lr{learning_rate}-dr{dropout_rate}.csv'
                    predictions_df.to_csv(rf'D:\Users\Alastor\Desktop\STAT216\STAT216V\Final\{file_name}', index=False)

# Process the entire dataset without separating by gender
X = feature_engineering(train_data.drop(columns=['id_num', 'political_affiliation']).values)
y = train_data['political_affiliation'].values

# Load the test data
X_test = test_data

# Define the parameter grid
param_grid = {
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 30, 50, 100],
    'learning_rate': [0.0001, 0.001, 0.01],
    'dropout_rate': [0.3, 0.5, 0.7]
}

# Manual hyperparameter tuning and saving predictions for all models
manual_grid_search(X, y, X_test, param_grid)

