from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# metric evaluation of classifiers
def evaluate_clf(clf, clf_name, X_train, X_test, y_train, y_test, sampling_tech='', df=None):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)

    entry = [clf_name, sampling_tech, accuracy, precision, recall, f1, fpr, tpr, roc_auc]
    if df is not None:
        df.loc[len(df)] = entry
    return y_pred, entry


def modelling(X_train, X_test, y_train, y_test):
    classifiers = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42, probability=True),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    sampling_techniques = {}
    sampling_techniques['base'] = {'X_train': X_train, 'y_train': y_train}
    
    # oversampling using RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)
    sampling_techniques['oversampling'] = {'X_train': X_train_oversampled, 'y_train': y_train_oversampled}
    
    # undersampling using RandomUnderSampler
    undersampler = RandomUnderSampler(random_state=42)
    X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)
    sampling_techniques['undersampling'] = {'X_train': X_train_undersampled, 'y_train': y_train_undersampled}

    results = pd.DataFrame(columns=['Model', 'SamplingTechnique', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR', 'TPR', 'AUC'])

    for name, clf in classifiers.items():
        # confusion matrix figure
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        fig.suptitle(f'Confusion Matrices for {name}', fontsize=16)

        for i, (method, train_data) in enumerate(sampling_techniques.items()):
            y_pred, _ = evaluate_clf(clf, name, train_data['X_train'], X_test, train_data['y_train'], y_test, method, df=results)

            # plot confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
            display.plot(ax=axs[i], cmap='Blues')
            axs[i].set_title(method) 
            
        plt.tight_layout()
        plt.show()
        print(results.loc[len(results)-len(sampling_techniques.keys()):len(results)-1].drop(['TPR', 'FPR'], axis=1).to_string())
    return results


# given data on models and different sampling techniques, compare the sampling techniques with their respective model
def sampling_comparison(data):
    classifiers = data['Model'].unique()
    
    fig_cols = 3
    fig_rows = math.ceil(len(classifiers) / fig_cols)
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(15, 10))

    # remove axis for extra plots
    if fig_rows > 1:
        for col in range(len(classifiers)%fig_cols, fig_cols):
            axs[fig_rows-1, col].axis('off')
        
    axs = axs.flatten()
    
    for ax, clf_name in zip(axs, classifiers):
        filtered = data[data['Model'] == clf_name]
        
        for technique in data['SamplingTechnique'].unique():
            entry = filtered[filtered['SamplingTechnique'] == technique]

            fpr = entry['FPR'].iloc[0]
            tpr = entry['TPR'].iloc[0]
            auc = entry['AUC'].iloc[0]

            ax.plot(fpr, tpr, label=f'{clf_name} {technique.capitalize()} (AUC = {auc:.2f})')
        
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve Comparison for {clf_name}')
        ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()
    return fig


# given data of models and different sampling techniques, compare each model with respect to the different sampling techniques
def model_comparison(data):
    sampling_techs = data['SamplingTechnique'].unique()
    
    fig_cols = 3
    fig_rows = math.ceil(len(sampling_techs) / fig_cols)
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(18, 6))

    # remove axis for extra plots (not applicable atm)
    if fig_rows > 1:
        for col in range(len(sampling_techs)%fig_cols, fig_cols):
            axs[fig_rows-1, col].axis('off')
    
    axs = axs.flatten()

    for ax, technique in zip(axs, sampling_techs):
        # filter data by sampling techniques
        filtered = data[data['SamplingTechnique'] == technique]
        
        for clf_name in data['Model'].unique():
            entry = filtered[filtered['Model'] == clf_name]

            fpr = entry['FPR'].iloc[0]
            tpr = entry['TPR'].iloc[0]
            auc = entry['AUC'].iloc[0]
            
            ax.plot(fpr, tpr, label=f'{clf_name} (AUC = {auc:.2f})')
        
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve Comparison ({technique.capitalize()})')
        ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()
    return fig


def main():
    dataset = fetch_ucirepo(id=468)
    print(dataset.metadata)


    features = dataset.data.features
    targets = dataset.data.targets

    data = pd.concat([features, targets], axis=1)
    print(data.info())

    corr = data.select_dtypes(include=[np.number, bool]).corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap='coolwarm', fmt='.2f', annot=True)
    plt.show()

    data.insert(loc=0, column='SessionDuration', value=data.Administrative_Duration+data.Informational_Duration+data.ProductRelated_Duration)
    data.insert(loc=1, column='AdministrativeAvgDuration', value=(data.Administrative_Duration/data.Administrative).replace(np.nan, 0))
    data.insert(loc=4, column='InformationalAvgDuration', value=(data.Informational_Duration/data.Informational).replace(np.nan, 0))
    data.insert(loc=7, column='ProductRelatedAvgDuration', value=(data.ProductRelated_Duration/data.ProductRelated).replace(np.nan, 0))
    print(data.isnull().sum())

    corr = data.select_dtypes(include=[np.number, bool]).corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', fmt='.2f', annot=True)
    plt.show()

    # identify highly correlated features (we'll set the threshold to 0.9)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # check for both highly pos and neg (some cases, neg extremes should be kept)
    high_corr = [col for col in upper.columns if any(upper[col].abs() > 0.9)]
    print(high_corr)

    data.drop(['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'ExitRates'], axis=1, inplace=True)

    corr = data.select_dtypes(include=[np.number, bool]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', fmt='.2f', annot=True)
    plt.show()

    # Data preprocessing
    # categorical encoding (one-hot encoding)
    data = pd.get_dummies(data, columns=['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType'], dtype=int, drop_first=True)

    # type conversion
    data[['Weekend', 'Revenue']] = data[['Weekend', 'Revenue']].astype(int)
    data.info()

    # feature scaling
    continuous_cols = data.select_dtypes(include=[float]).columns
    scaler = StandardScaler()
    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])

    # check for data imbalance
    print(data.Revenue.value_counts().to_string())

    neg, pos = np.bincount(data['Revenue'])
    total = neg + pos
    print(f"Minority class at {100*pos/total:.2f}% of total")

    X = data.drop('Revenue', axis=1)
    y = data['Revenue']

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Train set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    results = modelling(X_train, X_test, y_train, y_test)
    figure1 = sampling_comparison(results)
    figure2 = model_comparison(results)

    selector = SelectKBest(f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train.columns[selector.get_support()]
    print(selected_features.to_string())

    top10results = modelling(X_train_selected, X_test_selected, y_train, y_test)
    figure3 = sampling_comparison(top10results)
    figure4 = model_comparison(top10results)

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestClassifier(random_state=42)
    cv = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=5, scoring='f1', random_state=42)
    cv.fit(X_train, y_train)
    print(f"\nBest Random Forest Parameters: {cv.best_params_}")

    best_rf = cv.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    rf_f1 = f1_score(y_test, y_pred_rf)
    print("Best Random Forest F1 Score (on test set): {:.3f}".format(rf_f1))


if __name__ == "__main__":
    main()
