import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler # Import the scaler

# 1. Load and split the data
processed_data_file = 'new data.csv'
df = pd.read_csv(processed_data_file)
label_column = 'Bottleneck Feature 33'
X = df.drop(label_column, axis=1)
y = df[label_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Remove non-numerical columns (adjust as needed based on your data)
non_numerical_cols = ['59.166.0.0', '149.171.126.6', 'udp', 'dns', 'CON',
                       'Unnamed: 47']  # Add all non-numerical column names
print("Columns before dropping:", X_train.columns)
X_train = X_train.drop(non_numerical_cols, axis=1, errors='ignore')
X_test = X_test.drop(non_numerical_cols, axis=1, errors='ignore')

# Print columns after dropping
print("Columns after dropping:", X_train.columns)

# Convert y_test and y_train to binary
def convert_to_binary(labels):
    """
    Converts labels to binary: 0 for normal, 1 for anomaly.
    Adapt the normal class label as needed.
    """
    if isinstance(labels.iloc[0], str):  # Check if the first element is a string
        return [0 if x == 'normal' else 1 for x in labels]  # Adapt 'normal' if needed
    else:
        return [0 if x == 0 else 1 for x in labels]  # Assuming 0 is the normal class, change if needed


y_train_binary = convert_to_binary(y_train)
y_test_binary = convert_to_binary(y_test)



# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # scale test with the same parameters as training


# 2. Train the One-Class SVM (ASVM)
# Tune these parameters (nu, kernel) for your data
best_auc = 0
best_params = {}
# Refine nu for linear kernel
for nu in [0.15, 0.2, 0.25, 0.3]:
    print(f"Training with kernel=linear, nu={nu}")
    asvm = OneClassSVM(kernel='linear',
                       nu=nu)  # Important parameters to tune, fix kernel to linear
    asvm.fit(X_train_scaled) # Use the scaled data

    # 3. Predict anomalies
    y_train_pred = asvm.predict(X_train_scaled)
    y_test_pred = asvm.predict(X_test_scaled)

    # Convert predictions to binary (0 for normal, 1 for anomaly)
    y_train_pred_binary = [0 if x == 1 else 1 for x in y_train_pred]
    y_test_pred_binary = [0 if x == 1 else 1 for x in y_test_pred]

    # 4. Evaluate the ASVM
    print("Classification Report - Training Data:")
    print(classification_report(y_train_binary, y_train_pred_binary))

    print("\nClassification Report - Testing Data:")
    print(classification_report(y_test_binary, y_test_pred_binary))

    # Compute ROC AUC (if your y_test is binary)
    try:
        roc_auc = roc_auc_score(y_test_binary, y_test_pred_binary)
        print(f"\nROC AUC Score: {roc_auc:.2f}")

        # Plot ROC Curve for the *best* performing model
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_params['kernel'] = 'linear'
            best_params['nu'] = nu
            fpr, tpr, thresholds = roc_curve(y_test_binary,
                                            asvm.decision_function(
                                                X_test_scaled))  # Use decision_function with scaled data
    except ValueError:
        print(
            "\nROC AUC Score and ROC Curve not calculated.  y_test must be binary.")
    print("-" * 40)  # Separator for each run

print(f"Best parameters: kernel={best_params['kernel']}, nu={best_params['nu']}")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC AUC = {best_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()