import os
import time
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline

load_dotenv()
file_path = os.getenv("DATA_PATH")

def get_data(path = file_path) -> tuple[np.ndarray,np.ndarray,pd.Index]:
    df_raw = pd.read_csv(path)
    df_1 = df_raw.drop(columns=["id"])
    df = df_1.copy()
    feat_names = df.iloc[:,:-1].columns
    x_full = df.iloc[:,:-1]
    y_full = df.iloc[:,-1]
    x_sample,_,y_sample,_ = train_test_split(
        x_full,y_full,train_size=50_000,random_state=123,shuffle=True,stratify=y_full
    )
    return x_sample, y_sample, feat_names

def get_pipe() -> Pipeline:
    pipe_model = XGBClassifier(
        n_estimators=600,max_depth=10, learning_rate=0.1
    )
    pipe = Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("dimen",PCA(random_state=9987,n_components=24)),
        ("model",pipe_model)
    ])
    return pipe

def evaluate(est:BaseEstimator,x_test:np.ndarray,y_test:np.ndarray) -> None:
    y_true = y_test
    y_pred = est.predict(x_test)
    print("Classification Report:")
    print(classification_report(y_true,y_pred))

def main() -> None:
    x_sample, y_sample, feat_names = get_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x_sample,y_sample,test_size=0.2,random_state=432,shuffle=True,stratify=y_sample
    )
    pipe = get_pipe()

    print("Starting model fitting...")
    print("It may take a while, please sit tight...")
    t1 = time.time()
    pipe.fit(x_train,y_train)
    t2 = time.time()
    print("Model fitting completed successfully ✅")
    elapsed = t2 - t1
    mins,secs = np.divmod(elapsed,60)
    print(f"Time Elapsed: {mins:.0f} Minute {secs:.2f} Seconds")

    evaluate(pipe,x_test,y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe,"models/pipe.pkl")
    joblib.dump(feat_names,"models/feat_names.pkl")
    print("Models saved successfully ✅")


if __name__ == "__main__":
    main()

