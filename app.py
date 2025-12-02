import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DataProcessor:
    """Classe pour charger et nettoyer les données"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Charge les données depuis le fichier CSV"""
        self.df = pd.read_csv(self.filepath)
        print(f"Données chargées: {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
        return self.df

        
    def clean_data(self):
        """Nettoie les données"""
        # Supprimer les doublons
        self.df = self.df.drop_duplicates()
        print(f"\nDoublons supprimés")
        
        # Supprimer les lignes avec valeurs manquantes importantes
        self.df = self.df.dropna(subset=['Prix', 'Surface', 'Pièces', 'Gouvernorat'])
        
        # Supprimer les valeurs aberrantes
        self.df = self.df[
            (self.df['Prix'] > 0) & 
            (self.df['Surface'] > 0) & 
            (self.df['Pièces'] > 0) &
            (self.df['Prix'] < 10000) &
            (self.df['Surface'] < 500)
        ]
        
        # Créer de nouvelles colonnes
        self.df['prix_par_m2'] = self.df['Prix'] / self.df['Surface']
        self.df['surface_par_piece'] = self.df['Surface'] / self.df['Pièces']
        
        # Encoder le gouvernorat
        self.df['Gouvernorat_encoded'] = self.label_encoder.fit_transform(self.df['Gouvernorat'])
        
        print(f"Données après nettoyage: {self.df.shape[0]} lignes")
        return self.df


class PricePredictor:
    """Classe pour prédire les prix de location avec Régression Linéaire"""
    
    def __init__(self, features):
        self.features = features
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def prepare_data(self, df):
        """Prépare les données pour l'entraînement"""
        X = df[self.features].values
        y = df['Prix'].values
        
        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normaliser
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"\nDonnées divisées: {len(X_train)} entraînement, {len(X_test)} test")
        
    def train_model(self):
        """Entraîne le modèle de Régression Linéaire"""
        print("\n=== Entraînement du modèle ===")
        
        # Entraîner le modèle
        self.model.fit(self.X_train, self.y_train)
        
        # Faire des prédictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Calculer les performances sur l'ensemble d'entraînement
        r2_train = r2_score(self.y_train, y_pred_train)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        
        # Calculer les performances sur l'ensemble de test
        r2_test = r2_score(self.y_test, y_pred_test)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        
        print(f"\nRégression Linéaire:")
        print(f"  Performances sur l'entraînement:")
        print(f"    R²: {r2_train:.4f}")
        print(f"    MAE: {mae_train:.2f} TND")
        print(f"    RMSE: {rmse_train:.2f} TND")
        
        print(f"\n  Performances sur le test:")
        print(f"    R²: {r2_test:.4f}")
        print(f"    MAE: {mae_test:.2f} TND")
        print(f"    RMSE: {rmse_test:.2f} TND")
        
    def show_coefficients(self):
        """Affiche les coefficients du modèle"""
        print("\n=== Coefficients du modèle ===")
        for i, feature in enumerate(self.features):
            print(f"{feature}: {self.model.coef_[i]:.4f}")
        print(f"Intercept (constante): {self.model.intercept_:.4f}")
            
    def predict(self, features_array):
        """Fait une prédiction"""
        features_scaled = self.scaler.transform(features_array.reshape(1, -1))
        return self.model.predict(features_scaled)[0]


# ============= PROGRAMME PRINCIPAL =============

def main():
    """Fonction principale"""
    
    # 1. Charger et nettoyer les données
    processor = DataProcessor('annonces_appartements.csv')
    processor.load_data()
  
    df_clean = processor.clean_data()
    
    # 2. Définir les caractéristiques
    features = ['Pièces', 'Bains', 'Surface', 'Gouvernorat_encoded', 'surface_par_piece']
    
    # 3. Entraîner le modèle
    predictor = PricePredictor(features)
    predictor.prepare_data(df_clean)
    predictor.train_model()
    predictor.show_coefficients()
    
    # 4. Exemples de prédictions
    print("\n=== Exemples de prédictions ===")
    
    # Exemple 1: Appartement 3 pièces, 1 bain, 100m², Gouvernorat 8
    exemple1 = np.array([3, 1, 100, 8, 33.33])
    prix1 = predictor.predict(exemple1)
    print(f"\nAppartement 1: 3 pièces, 1 bain, 100m²")
    print(f"Prix prédit: {prix1:.2f} TND")
    
    # Exemple 2: Appartement 4 pièces, 2 bains, 150m², Gouvernorat 5
    exemple2 = np.array([4, 2, 150, 5, 37.5])
    prix2 = predictor.predict(exemple2)
    print(f"\nAppartement 2: 4 pièces, 2 bains, 150m²")
    print(f"Prix prédit: {prix2:.2f} TND")
    
    return df_clean, predictor


if __name__ == "__main__":
    df_clean, predictor = main()