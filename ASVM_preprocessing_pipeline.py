import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_csv(self, file_path):
        return pd.read_csv(file_path)

    def clean_data(self, df):
        return df.dropna()

    def encode_labels(self, df):
        categorical_columns = ['proto', 'state', 'service']
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
        return df

    def remove_features(self, df):
        columns_to_remove = ['srcip', 'dstip']
        return df.drop(columns=columns_to_remove, errors='ignore')

    def normalize_features(self, df):
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df

    def train_autoencoder(self, df):
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        input_dim = numeric_df.shape[1]
        encoding_dim = max(input_dim // 2, 1)
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        X = numeric_df.values
        autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, verbose=1)
        encoder = Model(input_layer, encoded)
        encoded_features = encoder.predict(X)
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=[f'Bottleneck Feature {i+1}' for i in range(encoding_dim)]
        )
        non_numeric_df = df.select_dtypes(exclude=['float64', 'int64'])
        return pd.concat([non_numeric_df.reset_index(drop=True),
                          numeric_df.reset_index(drop=True),
                          encoded_df.reset_index(drop=True)], axis=1)

    def save_csv(self, df, file_path):
        df.to_csv(file_path, index=False)