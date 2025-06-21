"""
Kerangka kerja komprehensif untuk peramalan runtun waktu konsumsi listrik.

Pipeline ini mencakup rekayasa fitur canggih, optimasi hiperparameter
dengan Optuna, dan ensemble dari berbagai model machine learning.
"""
import warnings
from datetime import datetime

import numpy as np
# PERBAIKAN: Ubah urutan impor untuk mengatasi Circular Import Error
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna

warnings.filterwarnings('ignore')


class ElectricityConsumptionPredictor:
    """
    Kelas utama untuk membungkus seluruh logika prediksi konsumsi listrik.
    """
    def __init__(self):
        """Inisialisasi semua atribut yang diperlukan."""
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_cols = []
        self.best_params = {}
        self.fallback_values = {}
        self.ensemble_weights = {}

    def create_advanced_features(self, df):
        """Menciptakan fitur-fitur turunan dari data mentah."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['cluster_id', 'date'])

        # Fitur berbasis waktu
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofyear'] = df['date'].dt.dayofyear
        df['weekday'] = df['date'].dt.weekday
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['season'] = ((df['month'] % 12 + 3) // 3).map({
            1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'
        })

        # Fitur siklus (cyclical features)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        # Fitur berbasis cuaca
        df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['temp_mean'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
        df['apparent_temp_mean'] = (
            (df['apparent_temperature_max'] + df['apparent_temperature_min']) / 2
        )
        df['temp_apparent_diff'] = df['temp_mean'] - df['apparent_temp_mean']
        df['heating_demand'] = np.maximum(0, 18 - df['temp_mean'])
        df['cooling_demand'] = np.maximum(0, df['temp_mean'] - 22)

        # Fitur interaksi & rasio
        df['wind_intensity'] = df['wind_gusts_10m_max'] / (df['wind_speed_10m_max'] + 1e-6)
        df['sunshine_ratio'] = df['sunshine_duration'] / (df['daylight_duration'] + 1e-6)

        # Fitur lag & rolling (autoregressive)
        if 'electricity_consumption' in df.columns:
            df['lag_1'] = df.groupby('cluster_id')['electricity_consumption'].shift(1)
            df['lag_7'] = df.groupby('cluster_id')['electricity_consumption'].shift(7)
            df['lag_14'] = df.groupby('cluster_id')['electricity_consumption'].shift(14)
            df['consumption_trend_7'] = df.groupby('cluster_id')['electricity_consumption'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )

        # Fitur rolling untuk cuaca
        for col in ['temp_mean', 'sunshine_duration', 'cooling_demand', 'heating_demand']:
            for win in [3, 7, 14]:
                df[f'{col}_ma_{win}'] = df.groupby('cluster_id')[col].transform(
                    lambda x, window=win: x.rolling(window, min_periods=1).mean()
                )
        return df

    def prepare_features(self, df, is_training=True):
        """Mempersiapkan fitur untuk training atau inference, termasuk encoding dan imputasi."""
        if 'temp_mean_ma_3' not in df.columns:
             df = self.create_advanced_features(df)

        categorical_cols = ['cluster_id', 'season']
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    known_labels = set(self.label_encoders[col].classes_)
                    new_labels = set(df[col].astype(str)) - known_labels
                    if new_labels:
                        self.label_encoders[col].classes_ = np.append(
                            self.label_encoders[col].classes_, sorted(list(new_labels))
                        )
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                else:
                    raise ValueError(f"Encoder untuk '{col}' tidak ditemukan.")

        if is_training:
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            self.feature_cols = [c for c in numerical_cols if c != 'electricity_consumption']

        x_df = df[self.feature_cols]

        if is_training:
            valid_rows_index = x_df.dropna().index
            x_df = x_df.loc[valid_rows_index]
            y_df = df.loc[valid_rows_index, 'electricity_consumption']
            self.fallback_values = x_df.mean().to_dict()
            return x_df, y_df
        
        x_df = x_df.fillna(method='ffill')
        x_df = x_df.fillna(value=self.fallback_values)
        if x_df.isnull().sum().sum() > 0:
            x_df = x_df.fillna(0)
        return x_df

    def optimize_hyperparameters(self, x_train, y_train, model_name, n_trials=30):
        """Optimasi hiperparameter untuk model yang diberikan menggunakan Optuna."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        objectives = {
            'lgb': self.lgb_objective,
            'xgb': self.xgb_objective,
            'svr': self.svr_objective
        }
        if model_name not in objectives:
            raise ValueError(f"Optimasi untuk model '{model_name}' tidak didukung.")

        study = optuna.create_study(direction='minimize')
        objective_func = objectives[model_name]

        if model_name == 'svr':
            print("  (SVR memerlukan penskalaan data untuk tuning...)")
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            study.optimize(lambda trial: objective_func(trial, x_train_scaled, y_train), n_trials=n_trials)
        else:
            study.optimize(lambda trial: objective_func(trial, x_train, y_train), n_trials=n_trials)

        self.best_params[model_name] = study.best_params
        return study.best_value

    def _get_objective_params(self, trial, model_name):
        """Helper untuk mendapatkan parameter trial Optuna berdasarkan nama model."""
        if model_name == 'lgb':
            return {
                'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                'max_depth': trial.suggest_int('max_depth', 5, 12),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
        if model_name == 'xgb':
            return {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-2, 1.0, log=True),
                'random_state': 42, 'n_jobs': -1
            }
        if model_name == 'svr':
            params = {
                'C': trial.suggest_float('C', 1e-1, 1e2, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-2, 1e0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
            }
            if params['kernel'] == 'rbf':
                params['gamma'] = trial.suggest_float('gamma', 1e-3, 1e-1, log=True)
            return params
        return {}

    def lgb_objective(self, trial, x_train, y_train):
        """Objective function untuk tuning LightGBM."""
        params = self._get_objective_params(trial, 'lgb')
        kfold = TimeSeriesSplit(n_splits=5)
        model = lgb.LGBMRegressor(**params)
        cv_scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        return np.sqrt(-cv_scores.mean())

    def xgb_objective(self, trial, x_train, y_train):
        """Objective function untuk tuning XGBoost."""
        params = self._get_objective_params(trial, 'xgb')
        kfold = TimeSeriesSplit(n_splits=5)
        model = xgb.XGBRegressor(**params)
        cv_scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        return np.sqrt(-cv_scores.mean())

    def svr_objective(self, trial, x_train_scaled, y_train):
        """Objective function untuk tuning SVR."""
        params = self._get_objective_params(trial, 'svr')
        kfold = TimeSeriesSplit(n_splits=5)
        model = SVR(**params)
        cv_scores = cross_val_score(model, x_train_scaled, y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        return np.sqrt(-cv_scores.mean())

    def train_ensemble(self, x_train, y_train, weighting_strategy='inverse_error', manual_weights=None):
        """Melatih semua model dalam ensemble dan menentukan bobotnya."""
        print(f"🏗️ Melatih ensemble dengan strategi: '{weighting_strategy}'...")

        self.scalers['standard'] = StandardScaler()
        x_train_scaled = self.scalers['standard'].fit_transform(x_train)
        x_train_scaled_df = pd.DataFrame(x_train_scaled, index=x_train.index, columns=x_train.columns)

        self.models['lgb'] = lgb.LGBMRegressor(**self.best_params.get('lgb', {}), random_state=42)
        self.models['xgb'] = xgb.XGBRegressor(**self.best_params.get('xgb', {}), random_state=42)
        self.models['svr'] = SVR(**self.best_params.get('svr', {}))
        self.models['cat'] = cb.CatBoostRegressor(random_state=42, verbose=0)
        self.models['et'] = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        self.models['ridge'] = Ridge(random_state=42)

        print("📊 Menghitung skor CV untuk pembobotan (semua model dievaluasi pada data terstandardisasi)...")
        cv_scores = {}
        kfold = TimeSeriesSplit(n_splits=5)
        for name, model in self.models.items():
            scores = cross_val_score(clone(model), x_train_scaled_df, y_train, cv=kfold,
                                     scoring='neg_mean_squared_error')
            cv_scores[name] = np.sqrt(-scores.mean())
            print(f"  - {name} CV RMSE: {cv_scores[name]:.4f}")

        if weighting_strategy == 'inverse_error':
            inv_errors = np.array([1 / (score + 1e-9) for score in cv_scores.values()])
            self.ensemble_weights = {name: w / inv_errors.sum() for name, w in zip(self.models, inv_errors)}
        elif weighting_strategy == 'equal':
            self.ensemble_weights = {name: 1.0 / len(self.models) for name in self.models}
        elif weighting_strategy == 'manual':
            if manual_weights is None or not isinstance(manual_weights, dict):
                raise ValueError("Bobot manual harus disediakan dalam format dictionary.")
            if abs(sum(manual_weights.values()) - 1.0) > 1e-6:
                raise ValueError("Jumlah bobot manual harus sama dengan 1.")
            self.ensemble_weights = manual_weights
        else:
            raise ValueError(f"Strategi pembobotan tidak dikenal: {weighting_strategy}")

        print("🏋️ Melatih model final pada seluruh data training...")
        for name, model in self.models.items():
            model.fit(x_train_scaled_df, y_train)

    def predict(self, x_test):
        """Melakukan prediksi pada data uji menggunakan ensemble yang sudah dilatih."""
        x_test_scaled = self.scalers['standard'].transform(x_test)
        ensemble_pred = np.zeros(len(x_test))

        for name, model in self.models.items():
            weight = self.ensemble_weights.get(name, 0)
            if weight > 0:
                prediction = model.predict(x_test_scaled)
                ensemble_pred += weight * prediction

        ensemble_pred[ensemble_pred < 0] = 0
        return ensemble_pred

    def plot_prediction_correlation(self, x_df):
        """Membuat heatmap korelasi antar prediksi dari setiap model."""
        print("\n📈 Membuat heatmap korelasi prediksi antar model...")
        x_scaled = self.scalers['standard'].transform(x_df)
        predictions_df = pd.DataFrame()
        for name, model in self.models.items():
            predictions_df[name] = model.predict(x_scaled)

        plt.figure(figsize=(10, 8))
        sns.heatmap(predictions_df.corr(), annot=True, cmap='coolwarm', fmt=".3f")
        plt.title('Heatmap Korelasi Prediksi Antar Model', fontsize=16)
        plt.show()

def main():
    """
    Fungsi utama untuk menjalankan keseluruhan pipeline.
    """
    print("📥 Memuat data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    if train_df['date'].max() >= test_df['date'].min():
        raise ValueError("❌ Tanggal test set tidak boleh tumpang tindih.")

    predictor = ElectricityConsumptionPredictor()

    print("🔗 Menggabungkan data train dan test untuk konsistensi fitur...")
    test_ids = test_df['ID']
    test_df['electricity_consumption'] = np.nan
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print("✨ Membuat fitur-fitur canggih...")
    full_df_featured = predictor.create_advanced_features(full_df)

    print("🪓 Memisahkan kembali data...")
    train_processed = full_df_featured[full_df_featured['electricity_consumption'].notna()].copy()
    test_processed = full_df_featured[full_df_featured['electricity_consumption'].isna()].copy()

    print("🧠 Mempersiapkan data training...")
    x_train, y_train = predictor.prepare_features(train_processed, is_training=True)

    print("🔧 Optimasi hyperparameter dimulai...")
    lgb_score = predictor.optimize_hyperparameters(x_train, y_train, 'lgb', n_trials=30)
    xgb_score = predictor.optimize_hyperparameters(x_train, y_train, 'xgb', n_trials=30)
    svr_score = predictor.optimize_hyperparameters(x_train, y_train, 'svr', n_trials=20)

    predictor.train_ensemble(x_train, y_train, weighting_strategy='inverse_error')

    print("📦 Mempersiapkan data test...")
    x_test = predictor.prepare_features(test_processed, is_training=False)

    print("🔮 Prediksi dimulai...")
    predictions = predictor.predict(x_test)

    print("📤 Membuat file submission...")
    submission = pd.DataFrame({'ID': test_ids, 'electricity_consumption': predictions})
    submission.to_csv('submission.csv', index=False)

    print("\n--- LAPORAN AKHIR ---")
    print(f"✅ LightGBM Optimized CV RMSE: {lgb_score:.4f}")
    print(f"✅ XGBoost Optimized CV RMSE: {xgb_score:.4f}")
    print(f"✅ SVR Optimized CV RMSE: {svr_score:.4f}")
    print("\nBobot Final Ensemble:")
    sorted_weights = sorted(predictor.ensemble_weights.items(), key=lambda item: item[1], reverse=True)
    for model, weight in sorted_weights:
        print(f"  - {model:<5}: {weight:.4f}")
    print("\n📁 File submission berhasil dibuat: submission.csv")

    predictor.plot_prediction_correlation(x_train)

if __name__ == "__main__":
    main()