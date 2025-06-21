import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone
import lightgbm as lgb
import xgboost as xgb
import optuna
warnings.filterwarnings('ignore')

class ElectricityConsumptionPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_cols = []
        self.best_params = {}
        self.fallback_values = {}

    def create_advanced_features(self, df, is_training=True):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['cluster_id', 'date'])

        if is_training and df.duplicated(['date', 'cluster_id']).any():
            raise ValueError("❌ Duplikasi ditemukan pada kombinasi (date, cluster_id)")

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofyear'] = df['date'].dt.dayofyear
        df['weekday'] = df['date'].dt.weekday
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['season'] = ((df['month'] % 12 + 3) // 3).map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

        df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['temp_mean'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
        df['apparent_temp_mean'] = (df['apparent_temperature_max'] + df['apparent_temperature_min']) / 2
        df['temp_apparent_diff'] = df['temp_mean'] - df['apparent_temp_mean']
        df['heating_demand'] = np.maximum(0, 20 - df['temp_mean'])
        df['cooling_demand'] = np.maximum(0, df['temp_mean'] - 24)

        df['wind_intensity'] = df['wind_gusts_10m_max'] / (df['wind_speed_10m_max'] + 1e-6)
        df['sunshine_ratio'] = df['sunshine_duration'] / (df['daylight_duration'] + 1e-6)

        if 'electricity_consumption' in df.columns:
            for cluster in df['cluster_id'].unique():
                mask = df['cluster_id'] == cluster
                if is_training:
                    df.loc[mask, 'lag_1'] = df.loc[mask, 'electricity_consumption'].shift(1)
                    df.loc[mask, 'lag_7'] = df.loc[mask, 'electricity_consumption'].shift(7)
                    df.loc[mask, 'lag_30'] = df.loc[mask, 'electricity_consumption'].shift(30)

                    df.loc[mask, 'consumption_trend_7'] = df.loc[mask, 'electricity_consumption'].rolling(window=7, min_periods=1).mean()
                    df.loc[mask, 'consumption_trend_30'] = df.loc[mask, 'electricity_consumption'].rolling(window=30, min_periods=1).mean()
                    df.loc[mask, 'consumption_trend_diff'] = df.loc[mask, 'electricity_consumption'] - df.loc[mask, 'consumption_trend_30']
                else:
                    for col in ['lag_1', 'lag_7', 'lag_30', 'consumption_trend_7', 'consumption_trend_30', 'consumption_trend_diff']:
                        df.loc[mask, col] = np.nan

        for col in ['temp_mean', 'sunshine_duration', 'cooling_demand', 'heating_demand']:
            for win in [3, 7, 14]:
                df[f'{col}_ma_{win}'] = df.groupby('cluster_id')[col].transform(lambda x: x.rolling(win, min_periods=1).mean())

        return df


    def prepare_features(self, df, is_training=True):
        required_cols = ['date', 'temperature_2m_max', 'temperature_2m_min', 
                        'apparent_temperature_max', 'apparent_temperature_min', 
                        'sunshine_duration', 'cluster_id']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"❌ Kolom wajib hilang dari data: {missing}")

        df = self.create_advanced_features(df, is_training=is_training)

        categorical_cols = ['cluster_id', 'season']
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                if col in self.label_encoders:
                    df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
                else:
                    raise ValueError(f"Encoder untuk '{col}' tidak ditemukan saat inference.")

        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
        feature_candidates = list(set(numerical_cols + encoded_cols) - {'electricity_consumption'})

        if is_training:
            self.feature_cols = feature_candidates

        X = df[self.feature_cols]

        if is_training:
            valid_rows = X.dropna().index
            X = X.loc[valid_rows]
            y = df.loc[valid_rows, 'electricity_consumption']
            self.fallback_values = X.mean().to_dict()
            return X, y
        else:
            X = X.fillna(method='ffill')
            X = X.fillna(value=self.fallback_values)
            return X


    def lgb_objective(self, trial, x_train, y_train):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 1000, 2500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 15),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 1e1, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        kfold = TimeSeriesSplit(n_splits=5)
        model = lgb.LGBMRegressor(**params)
        cv_scores = cross_val_score(model, x_train, y_train, cv=kfold, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        return np.sqrt(-cv_scores.mean())

    def xgb_objective(self, trial, x_train, y_train):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 1000, 2500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
            'random_state': 42,
            'n_jobs': -1
        }
        
        kfold = TimeSeriesSplit(n_splits=5)
        model = xgb.XGBRegressor(**params)
        cv_scores = cross_val_score(model, x_train, y_train, cv=kfold, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        return np.sqrt(-cv_scores.mean())

    def optimize_hyperparameters(self, x_train, y_train, model_name='lgb', n_trials=100):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if model_name == 'lgb':
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.lgb_objective(trial, x_train, y_train), n_trials=n_trials)
            self.best_params['lgb'] = study.best_params
            return study.best_value

        elif model_name == 'xgb':
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self.xgb_objective(trial, x_train, y_train), n_trials=n_trials)
            self.best_params['xgb'] = study.best_params
            return study.best_value

        else:
            raise ValueError(f"Model {model_name} belum didukung untuk tuning.")


    def train_ensemble(self, x_train, y_train):
        self.scalers['standard'] = StandardScaler()
        x_train_scaled = self.scalers['standard'].fit_transform(x_train)

        lgb_params = self.best_params['lgb'].copy()
        lgb_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        })
        self.models['lgb'] = lgb.LGBMRegressor(**lgb_params)

        xgb_params = self.best_params['xgb'].copy()
        xgb_params.update({
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        })
        self.models['xgb'] = xgb.XGBRegressor(**xgb_params)

        self.models['rf'] = RandomForestRegressor(
            n_estimators=600, max_depth=14, min_samples_split=4, min_samples_leaf=2,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=600, learning_rate=0.025, max_depth=6, subsample=0.85,
            min_samples_split=4, min_samples_leaf=2, random_state=42
        )
        self.models['ridge'] = Ridge(alpha=0.5, random_state=42)
        self.models['et'] = ExtraTreesRegressor(
            n_estimators=400, max_depth=12, min_samples_split=5, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )

        cv_scores = {}
        kfold = TimeSeriesSplit(n_splits=5)

        for name, model in self.models.items():
            x_input = x_train_scaled if name == 'ridge' else x_train
            scores = cross_val_score(clone(model), x_input, y_train, cv=kfold,
                                    scoring='neg_mean_squared_error', n_jobs=-1)
            cv_scores[name] = -scores.mean()

        for name, model in self.models.items():
            x_input = x_train_scaled if name == 'ridge' else x_train
            model.fit(x_input, y_train)

        inv_errors = np.array([1 / s for s in cv_scores.values()])
        weights = inv_errors / inv_errors.sum()
        self.ensemble_weights = {name: w for name, w in zip(cv_scores.keys(), weights)}

        return cv_scores


    def predict(self, X_test):
        # Transformasi menggunakan skala yang sama seperti saat training
        X_test_scaled = self.scalers['standard'].transform(X_test)

        # Menampung prediksi dari semua model
        predictions = {}
        for name, model in self.models.items():
            x_input = X_test_scaled if name == 'ridge' else X_test
            predictions[name] = model.predict(x_input)

        # Ensemble: rata-rata tertimbang
        ensemble_pred = np.zeros(len(X_test))
        for name, pred in predictions.items():
            ensemble_pred += self.ensemble_weights.get(name, 0) * pred

        # Validasi hasil akhir
        if np.isnan(ensemble_pred).any():
            raise ValueError("❌ Prediksi mengandung NaN. Periksa alur preprocessing.")
        if (ensemble_pred < 0).any():
            print("⚠️ Peringatan: Prediksi mengandung nilai negatif.")

        return ensemble_pred

    def get_feature_importance(self, model_name='lgb'):
        if model_name in self.models and hasattr(self.models[model_name], 'feature_importances_'):
            return pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.models[model_name].feature_importances_
            }).sort_values(by='importance', ascending=False)
        return None

def main():
    # 1. Load data
    print("📥 Memuat data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # 2. Konversi & Validasi Waktu
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    if train_df['date'].max() >= test_df['date'].min():
        raise ValueError("❌ Tanggal test set tidak boleh tumpang tindih atau lebih awal dari training set.")

    # 3. Inisialisasi Predictor
    predictor = ElectricityConsumptionPredictor()

    # 4. Arsitektur "Gold Standard": Gabungkan data untuk konsistensi fitur
    print("🔗 Menggabungkan data train dan test untuk konsistensi fitur...")
    # Simpan ID test set untuk submission nanti
    test_ids = test_df['ID']
    # Tambahkan placeholder NaN untuk digabungkan
    test_df['electricity_consumption'] = np.nan
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # 5. Buat semua fitur pada data gabungan. 
    print("✨ Membuat fitur-fitur canggih...")
    full_df_featured = predictor.create_advanced_features(full_df)

    # 6. Pisahkan kembali menjadi train dan test berdasarkan keberadaan target
    print("🪓 Memisahkan kembali data train dan test...")
    train_processed = full_df_featured[full_df_featured['electricity_consumption'].notna()].copy()
    test_processed = full_df_featured[full_df_featured['electricity_consumption'].isna()].copy()

    # 7. Siapkan data training: bersihkan NaN dan dapatkan (X, y) yang selaras
    print("🧠 Mempersiapkan data training...")
    x_train, y_train = predictor.prepare_features(train_processed, is_training=True)

    # 8. Optimasi hyperparameter
    print("🔧 Optimasi hyperparameter dimulai...")
    lgb_score = predictor.optimize_hyperparameters(x_train, y_train, model_name='lgb', n_trials=50)
    xgb_score = predictor.optimize_hyperparameters(x_train, y_train, model_name='xgb', n_trials=50)

    # 9. Latih model ensemble
    print("🏗️ Melatih ensemble...")
    predictor.train_ensemble(x_train, y_train)

    # 10. Siapkan data test
    print("📦 Mempersiapkan data test...")
    X_test = predictor.prepare_features(test_processed, is_training=False)

    # 11. Prediksi
    print("🔮 Prediksi dimulai...")
    predictions = predictor.predict(X_test)

    # 12. Simpan hasil
    print("📤 Membuat file submission...")
    submission = pd.DataFrame({'ID': test_ids, 'electricity_consumption': predictions})
    submission.to_csv('submission.csv', index=False)

    # 13. Laporan akhir
    print("\n--- LAPORAN AKHIR ---")
    print(f"✅ LightGBM Optimized CV RMSE: {lgb_score:.4f}")
    print(f"✅ XGBoost Optimized CV RMSE: {xgb_score:.4f}")
    print("\nBobot Final Ensemble:")
    for model, weight in sorted(predictor.ensemble_weights.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {model:<5}: {weight:.4f}")
    print("\n📁 File submission berhasil dibuat: submission.csv")


if __name__ == "__main__":
    main()
