from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from sklearn.metrics import r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

# Vercel's filesystem is read-only except /tmp.
# Use /tmp/artifacts for writing model files on Vercel.
IS_VERCEL = os.environ.get("VERCEL", False)
ARTIFACTS_DIR = os.path.join("/tmp", "artifacts") if IS_VERCEL else os.path.join(BASE_DIR, "artifacts")

def save_object(file_path, target_obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(target_obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, Y_train, X_test, Y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv = 3)
            gs.fit(X_train, Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, Y_train)

            Y_test_pred = model.predict(X_test)

            test_model_score = r2_score(Y_test, Y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)