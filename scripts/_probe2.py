import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from scipy.special import softmax

X, y = make_classification(n_samples=300, n_classes=3, n_informative=5, random_state=42)

# Global model is a list of boosters
global_model = []

# Initial margins (0 for all classes)
margins_1 = np.zeros((100, 3))
margins_2 = np.zeros((100, 3))
margins_3 = np.zeros((100, 3))

X1, y1 = X[:100], y[:100]
X2, y2 = X[100:200], y[100:200]
X3, y3 = X[200:], y[200:]

for round_idx in range(5):
    # Train client 1
    dtrain1 = xgb.DMatrix(X1, label=y1)
    dtrain1.set_base_margin(margins_1.flatten())
    b1 = xgb.train({"objective": "multi:softprob", "num_class": 3, "tree_method": "hist", "max_depth": 3}, dtrain1, num_boost_round=5)
    
    # Train client 2
    dtrain2 = xgb.DMatrix(X2, label=y2)
    dtrain2.set_base_margin(margins_2.flatten())
    b2 = xgb.train({"objective": "multi:softprob", "num_class": 3, "tree_method": "hist", "max_depth": 3}, dtrain2, num_boost_round=5)
    
    # Train client 3
    dtrain3 = xgb.DMatrix(X3, label=y3)
    dtrain3.set_base_margin(margins_3.flatten())
    b3 = xgb.train({"objective": "multi:softprob", "num_class": 3, "tree_method": "hist", "max_depth": 3}, dtrain3, num_boost_round=5)
    
    global_model.append([b1, b2, b3])
    
    # Update local margins for next round using the NEW boosters
    # Weight of each client's update is 1/3
    # Wait, the booster output margin is the sum of base_margin + new_trees_margin.
    # To get ONLY the new_trees_margin, we predict with output_margin=True and subtract base_margin?
    # No, predict(DMatrix) without base_margin gives the trees' margin!
    def get_trees_margin(b, X_data):
        return b.predict(xgb.DMatrix(X_data), output_margin=True)
    
    # Client 1 updates its margin with b1, b2, b3
    margins_1 += (get_trees_margin(b1, X1) + get_trees_margin(b2, X1) + get_trees_margin(b3, X1)) / 3.0
    margins_2 += (get_trees_margin(b1, X2) + get_trees_margin(b2, X2) + get_trees_margin(b3, X2)) / 3.0
    margins_3 += (get_trees_margin(b1, X3) + get_trees_margin(b2, X3) + get_trees_margin(b3, X3)) / 3.0

# Evaluate global model
def predict_global(X_data):
    total_margin = np.zeros((len(X_data), 3))
    for round_boosters in global_model:
        round_margin = np.zeros((len(X_data), 3))
        for b in round_boosters:
            round_margin += get_trees_margin(b, X_data)
        total_margin += round_margin / len(round_boosters)
    return softmax(total_margin, axis=1)

preds = np.argmax(predict_global(X), axis=1)
print("Accuracy:", accuracy_score(y, preds))
