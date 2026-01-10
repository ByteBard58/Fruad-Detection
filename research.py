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

    In this project, we will try to build a robust ML classifier that will be trained on a transaction record dataset (retrieved from Kaggle). This classifier will be able to predict a transaction's validity properly.

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
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier
    from xgboost import XGBClassifier

    from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,learning_curve,train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.metrics import confusion_matrix, classification_report

    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    return (
        BaggingClassifier,
        LDA,
        LogisticRegression,
        PCA,
        Pipeline,
        RandomForestClassifier,
        RandomizedSearchCV,
        SVC,
        SimpleImputer,
        StandardScaler,
        XGBClassifier,
        classification_report,
        confusion_matrix,
        learning_curve,
        np,
        pd,
        plt,
        sns,
        time,
        train_test_split,
    )


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
    return (feat_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Input-output separation and dataset sampling
    """)
    return


@app.cell
def _(df):
    x_full = df.iloc[:,:-1]
    y_full = df.iloc[:,-1]
    return x_full, y_full


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use `train_test_split` to get a smaller dataset
    """)
    return


@app.cell
def _(train_test_split, x_full, y_full):
    x_sample,_,y_sample,_ = train_test_split(
        x_full,y_full,train_size=50_000,random_state=123,shuffle=True,stratify=y_full
    )
    return x_sample, y_sample


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check class proportion
    """)
    return


@app.cell
def _(y_sample):
    y_sample.value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It seems like the class proportion is as expected. Now, let's move on to EDA.
    """)
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
    The countplot reveals perfect class balance between **Class 1 (Fraudulent)** and **Class 0 (Normal Transaction)**. It means there is no need for SMOTE (Synthetic Minority Over-sampling Technique) in our pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Note on Class Balance
    This dataset is artificially balanced for research purposes. In real-world fraud detection, fraudulent transactions are extremely rare, and techniques such as cost-sensitive learning or resampling would be mandatory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Machine Learning Part
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Final Split
    """)
    return


@app.cell
def _(train_test_split, x_sample, y_sample):
    x_train, x_test, y_train, y_test = train_test_split(
        x_sample,y_sample,test_size=0.2,random_state=432,shuffle=True,stratify=y_sample
    )
    return x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define Pipeline
    """)
    return


@app.cell
def _(
    BaggingClassifier,
    LDA,
    LogisticRegression,
    PCA,
    Pipeline,
    RandomForestClassifier,
    SVC,
    SimpleImputer,
    StandardScaler,
    XGBClassifier,
):
    pipe = Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("dimen",PCA(random_state=9987)),
        ("model",RandomForestClassifier(random_state=10243))
    ])

    rf = RandomForestClassifier(class_weight="balanced",random_state=32)
    lr = LogisticRegression(class_weight="balanced",random_state=43)
    svc = SVC(class_weight="balanced",random_state=78)
    xgb = XGBClassifier(random_state=902)
    bagclf = BaggingClassifier(random_state=1934)

    lda = LDA(n_components=1)
    pca = PCA(random_state=99877)
    return lda, pca, pipe, rf, svc, xgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Randomized Search
    """)
    return


@app.cell
def _(lda, pca, rf, svc, xgb):
    param_grid = [
        { ## RF, NO DIMEN
            "model":[rf], "model__n_estimators":[700,900,1200], "model__max_depth":[7,10,15], 
            "dimen": ["passthrough"]
        },
        { ## RF, LDA
            "model":[rf], "model__n_estimators":[700,900,1200], "model__max_depth":[7,10,15], 
            "dimen": [lda]
        },
        { ## RF, PCA
            "model":[rf], "model__n_estimators":[700,900,1200], "model__max_depth":[7,10,15], 
            "dimen":[pca],"dimen__n_components": [18,24,28]
        },
        { ## XGB, NO DIMEN
            "model":[xgb], "model__n_estimators":[600,900], "model__max_depth":[7,10,15],
            "dimen": ["passthrough"], "model__learning_rate": [0.01,0.1]
        },
        { ## XGB, PCA
            "model":[xgb], "model__n_estimators":[600,900], "model__max_depth":[7,10,15],
            "dimen":[pca],"dimen__n_components": [18,24,28], "model__learning_rate": [0.01,0.1]
        },
        { ## SVC, NO DIMEN
            "model": [svc], "model__C":[0.01,0.1,1,10,100], "model__kernel":["rbf"],"model__gamma":[0.1,0.5,1],
            "dimen": ["passthrough"]
        }
    ]
    return (param_grid,)


@app.cell
def _(RandomizedSearchCV, param_grid, pipe):
    rscv = RandomizedSearchCV(
        param_distributions=param_grid,estimator=pipe,
        cv=5,n_iter=10,n_jobs=3,
        random_state=31945,refit=True,verbose=1
    )
    return (rscv,)


@app.cell
def _(np, rscv, time, x_train, y_train):
    t1 = time.time()
    rscv.fit(x_train,y_train)
    t2 = time.time()
    elapsed = t2 - t1
    minutes, seconds = np.divmod(elapsed,60)
    print(f"Time Elapsed: {minutes} Minute {seconds} Seconds")

    est = rscv.best_estimator_
    scr = rscv.best_score_
    config = rscv.best_params_
    print(f"Best Score = {scr}")
    print(f"Best Configuration;\n{config}")
    return (est,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok, that is truly wonderful! **XGBClassifier (Extreme Gradient Boost)** has yielded a **near-perfect score (≈0.99)** with **PCA** feature extraction turned on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **NOTE:** As we all know, accuracy is often considered a useless metric. It doesn't focus on the significant details, which are only visible in the real space. To truly evaluate the model's robustness, we need to look at some other metrics like precision and recall.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Confusion Matrix
    Let's take a look at the confusion matrix of the model, which will help us to analyze the model's performance in reality.
    """)
    return


@app.cell
def _(confusion_matrix, est, x_test, y_test):
    y_true = y_test
    y_pred = est.predict(x_test)
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    return conf_matrix, y_pred, y_true


@app.cell
def _(conf_matrix, plt, sns):
    classes = [0,1]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='flare', 
                xticklabels=classes, yticklabels=classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alright. So the confusion matrix plot shows us that there are **no False Negatives** in our test, which means no fraudulent transaction should get through our model undetected. However, there are a few false positives, which may cause some manual review; the probability appears to be pretty low.

    Now, let's take a look at the **classification report**, which will let us check the metrics in one glance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Classification Report
    """)
    return


@app.cell
def _(classification_report, y_pred, y_true):
    print(classification_report(y_true,y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The classification report is indicating that the model is perfect from every metric's POV. Now, we can clearly declare this model as good-to-go without just relying on the accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PCA Loading
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loading Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's do some analysis on the PCA that is performed in the model.
    """)
    return


@app.cell
def _(est):
    pca_est = est.named_steps["dimen"]
    comps = pca_est.components_.T  # Components
    exp_var = pca_est.explained_variance_  # Explained Variance
    return comps, exp_var


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we know,
    $$\text{Loadings} = \text{pca.components\_}^T \times \sqrt{\text{pca.explained\_variance\_}}$$
    """)
    return


@app.cell
def _(comps, exp_var, np):
    loadings = comps * np.sqrt(exp_var)  ## Raw Loading
    return (loadings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Loading DataFrame
    """)
    return


@app.cell
def _(comps, feat_names, loadings, np, pd):
    sz = np.size(comps,axis=1)
    loading_df = pd.DataFrame(data=loadings, columns=[f"PC{n}" for n in range(sz)], index=feat_names)
    return (loading_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loading Heatmap
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's visualize it with a heatmap
    """)
    return


@app.cell
def _(loading_df, plt, sns):
    plt.figure(figsize=(11,7))
    sns.heatmap(loading_df,annot=True,cmap="icefire",center=0,fmt=".1f")
    plt.title("PCA Loading Heatmap", fontdict={"fontsize":15})
    plt.xlabel("Features", fontdict={"fontsize":13})
    plt.ylabel("Features", fontdict={"fontsize":13})
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PC0 acts as a general factor capturing the core variance of most V-features, while subsequent components isolate unique relationships like the strong influence of V21 on PC1. The "Amount" variable is statistically independent of the primary features, influencing only specialized dimensions like PC6 and PC7 rather than the main variance. The high concentration of strong loadings in the initial components proves that the 29 original variables are driven by a few dominant underlying patterns.

    However, due to feature anonymization, PCA interpretation remains statistical rather than semantic.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learning Curve
    """)
    return


@app.cell
def _(est, learning_curve, np, x_train, y_train):
    train_size, train_score, val_score = learning_curve(
        est,x_train,y_train, train_sizes=np.linspace(0.1,1.0,10),
        cv=5, shuffle=True,random_state=291492,n_jobs=3
    )
    return train_score, train_size, val_score


@app.cell
def _(np, plt, train_score, train_size, val_score):
    train_mean = np.mean(train_score, axis=1)
    train_std = np.std(train_score,axis=1)
    val_mean = np.mean(val_score,axis=1)
    val_std = np.std(val_score,axis=1)

    plt.figure(figsize=(10,6))
    plt.plot(train_size, train_mean, color="red",marker="s",markersize=4,label="Training Accuracy")
    plt.fill_between(train_size, train_mean + train_std , train_mean - train_std, color="red",alpha=0.3)

    plt.plot(train_size, val_mean, color="orange",marker="v",markersize=4,label="Validation Accuracy")
    plt.fill_between(train_size, val_mean + val_std, val_mean - val_std, color="orange",alpha=0.3)

    plt.title("Learning Curve (Random Forest with PCA)",fontdict={"fontsize":16})
    plt.xlabel("Train Size",fontdict={"fontsize":13})
    plt.ylabel("Accuracy",fontdict={"fontsize":13})
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The learning curve shows that the model is already very robust. The validation accuracy is almost the same as the training accuracy. Though the tiny gap between the two lines means there is a bit of an overfitting issue, it can be easily fixed by adding more data.

    In this case, that is very simple. Because we have only worked with a tiny sample of 50,000 rows of data from the original Kaggle dataset, which has more than 500,000 rows of transaction data. By adding more data to the sample, we may also yield a perfect 1.00 score. Due to sheer computational cost, I am going to avoid that for now.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook, we have successfully designed a near-perfect ML model to classify transactions as fraudulent or valid. We also looked at some important plots, which helped us to make more sense of the model and the data itself.

    We have chosen **XGBClassifier** with **PCA** feature extraction as our primary classifier because of its leading performance in `RandomizedSearchCV`. The feature extraction has helped to reduce complexity in the data and contributed to the overall robustness of the model. Instead of relying just on accuracy score, we evaluated other metrics like precision, recall and f1-score, which indicates that the model is truly robust for real-life implications.

    **Thank you** for taking your time to review this notebook. I hope you enjoyed it. If you have any comments or feedback, please share it with me. The code from this notebook will be used in other aspects of the project.
    """)
    return


if __name__ == "__main__":
    app.run()
