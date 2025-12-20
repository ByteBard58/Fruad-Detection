import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Transaction Fraud Detection

    In this project, I will try to build a robust ML classifier which will be trained on a transaction record dataset (retrieved from Kaggle). This classifier will be able to predict a transaction's validity properly.

    This notebook will be used as a sandbox to build the classifier from the ground up.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import libraries
    """)
    return


@app.cell
def _():
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from xgboost import XGBClassifier

    from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,learning_curve
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    return pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import dataset
    """)
    return


@app.cell
def _(pd):
    df_raw = pd.read_csv("Dataset/creditcard_2023.csv")
    return (df_raw,)


@app.cell
def _(df_raw):
    df_1 = df_raw.drop(columns=["id"],axis=0)
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check for Null
    """)
    return


@app.cell
def _(df_1):
    df_1.isna().value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No empty value detected. We can skip handling them for now.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Copy `df_1` to `df`
    """)
    return


@app.cell
def _(df_1):
    df = df_1.copy()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save Column Names
    """)
    return


@app.cell
def _(df):
    feat_names = df.iloc[:,:-1].columns
    all_names = df.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA (Exploratory Data Analysis)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feature Correlation Heatmap
    """)
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(16,8))
    cor_df = df.iloc[:,:-1].corr()
    sns.heatmap(cor_df,annot=True,fmt=".1f",cmap="icefire")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The heatmap reveals that only a few features exhibit a relatively strong correlation. Features from V1 to V18 show an intermediate to high correlation, while features from V19 to Amount exhibit almost no correlation among themselves. This suggests that dimensionality reduction should be a crucial component of the pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Class Distribution
    """)
    return


@app.cell
def _(df, plt, sns):
    sns.countplot(x="Class",data=df)
    plt.title("Distribution of Classes")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The countplot reveals perfect class balance between **Class 1(Fraudulent)** and **Class 0(Normal Transaction)**. It means there is not much need for SMOTE(Synthetic Minority Over-sampling Technique) in our pipeline.
    """)
    return


if __name__ == "__main__":
    app.run()
