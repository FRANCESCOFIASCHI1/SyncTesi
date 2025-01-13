import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, average_precision_score
from pyod.utils.data import precision_n_scores
from pyod.models.iforest import IForest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
# Per l'uso della memoria degli algoritmi
from memory_profiler import memory_usage
# Per la metrica sul tempo di Addestramento e Inferenza
import time

def evaluate_metrics(y_test, y_pred, y_proba=None, digits=3):
    res = {"Accuracy": round(accuracy_score(y_test, y_pred), digits),
           "Precision": precision_score(y_test, y_pred).round(digits),
           "Recall": recall_score(y_test, y_pred).round(digits),
           "F1": f1_score(y_test, y_pred).round(digits),
           "MCC": round(matthews_corrcoef(y_test, y_pred), ndigits=digits)}
    if y_proba is not None:
        res["AUC_PR"] = average_precision_score(y_test, y_proba).round(digits)
        res["AUC_ROC"] = roc_auc_score(y_test, y_proba).round(digits)
        res["PREC_N_SCORES"] = precision_n_scores(y_test, y_proba).round(digits)
    return res