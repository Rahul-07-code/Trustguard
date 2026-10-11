import numpy as np
from sklearn.ensemble import IsolationForest
import xgboost as xgb

class UpdateAnomalyDetector:
    def __init__(self, contamination=0.05, random_state=42):
        self.iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        self.is_fitted = False
        self.history = []

    def extract_features(self, booster: xgb.Booster, num_features: int) -> np.ndarray:
        # Extract feature importance (gain) as a fixed-size vector
        scores = booster.get_score(importance_type='gain')
        # Features are usually named f0, f1, ... fN
        vec = np.zeros(num_features)
        for k, v in scores.items():
            if k.startswith('f'):
                try:
                    idx = int(k[1:])
                    if idx < num_features:
                        vec[idx] = v
                except:
                    pass
            else:
                # if named features
                pass 
                
        # If we just get empty, use a fallback
        if len(scores) > 0:
            return vec / (np.sum(vec) + 1e-9)
        return vec

    def fit_or_evaluate(self, booster: xgb.Booster, num_features: int) -> tuple[bool, float]:
        vec = self.extract_features(booster, num_features)
        
        if not self.is_fitted:
            self.history.append(vec)
            if len(self.history) > 3:
                self.iso_forest.fit(np.array(self.history))
                self.is_fitted = True
            return True, 1.0 # 1 means normal
        else:
            # Evaluate FIRST
            score = self.iso_forest.decision_function([vec])[0]
            prediction = self.iso_forest.predict([vec])[0]
            is_normal = (prediction == 1)
            
            # Only add normal updates to history
            if is_normal:
                self.history.append(vec)
                self.iso_forest.fit(np.array(self.history))
                
            return is_normal, float(score)
