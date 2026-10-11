import json
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_classes=3, n_informative=5, random_state=42)

# Train model 1
model1 = xgb.XGBClassifier(n_estimators=2, max_depth=2, learning_rate=0.1)
model1.fit(X, y)
model1.save_model("m1.json")

# Train model 2
model2 = xgb.XGBClassifier(n_estimators=2, max_depth=2, learning_rate=0.1)
model2.fit(X, y)
model2.save_model("m2.json")

def merge_models(m1_path, m2_path, out_path):
    with open(m1_path) as f:
        d1 = json.load(f)
    with open(m2_path) as f:
        d2 = json.load(f)
        
    trees1 = d1["learner"]["gradient_booster"]["model"]["trees"]
    trees2 = d2["learner"]["gradient_booster"]["model"]["trees"]
    
    # Scale leaf weights by 0.5
    for t in trees1 + trees2:
        if "base_weights" in t:
            t["base_weights"] = [w * 0.5 for w in t["base_weights"]]
            
    d1["learner"]["gradient_booster"]["model"]["trees"] = trees1 + trees2
    d1["learner"]["gradient_booster"]["model"]["tree_info"] = d1["learner"]["gradient_booster"]["model"]["tree_info"] + d2["learner"]["gradient_booster"]["model"]["tree_info"]
    
    # Update num_trees
    d1["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(len(trees1 + trees2))
    
    # Update iteration_indptr
    # iteration_indptr tells xgboost which trees belong to which iteration
    # Original for 2 trees, 3 classes: [0, 3, 6]
    # We merged 2 models, so we have 4 iterations total: [0, 3, 6, 9, 12]
    num_class = 3
    num_iterations_total = len(trees1 + trees2) // num_class
    d1["learner"]["gradient_booster"]["model"]["iteration_indptr"] = [i * num_class for i in range(num_iterations_total + 1)]
    
    with open(out_path, "w") as f:
        json.dump(d1, f)

merge_models("m1.json", "m2.json", "m_merged.json")

model_merged = xgb.XGBClassifier()
model_merged.load_model("m_merged.json")
print("Merged model predicts:", model_merged.predict(X[:5]))
