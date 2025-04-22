import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

class Preprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.pca = None

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

    def apply_pca(self, df, n_components=5):
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(numeric_df)
        pca_df = pd.DataFrame(
            pca_features, 
            columns=[f'Principal Component {i+1}' for i in range(n_components)]
        )
        non_numeric_df = df.select_dtypes(exclude=['float64', 'int64'])
        return pd.concat([non_numeric_df.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

    def save_csv(self, df, file_path):
        df.to_csv(file_path, index=False)