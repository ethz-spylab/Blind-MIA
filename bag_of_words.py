from statistics import mean, pstdev
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from utils import *


def train_classifier(X_train, X_test, y_train, y_test, fpr_budget, plot_roc, 
                     params={'max_features': 30, 'vectorizer': 'tf', 'model_type': 'stack'}):

    if params['vectorizer'] == 'tf':
        converter = TfidfVectorizer(max_features=params['max_features'])
    elif params['vectorizer'] == 'count':
        converter = CountVectorizer(max_features=params['max_features'])

    X_train_Tfidf_df = converter.fit_transform(X_train).toarray()
    X_train_Tfidf_df = pd.DataFrame(X_train_Tfidf_df)
    X_test_Tfidf_df = converter.transform(X_test).toarray()
    X_test_Tfidf_df = pd.DataFrame(X_test_Tfidf_df)

    # print("Fiting model...")
    
    if params['model_type'] == 'random':
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    elif params['model_type'] == 'gaussian':
        model = GaussianNB()
    elif params['model_type'] == 'multi':
        model = MultinomialNB()
    elif params['model_type'] == 'gradient':
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=42)
    elif params['model_type'] == 'stack':
        estimators = [
                ('random', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
                ('gaussian', GaussianNB()),
                ('multi', MultinomialNB()),
                ('gradient', GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=5, random_state=42))
                ]
        model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42))

    model.fit(X_train_Tfidf_df, y_train)
    # y_pred = model.predict(X_test_Tfidf_df)
    y_pred_proba = model.predict_proba(X_test_Tfidf_df)[:,1]

    roc_auc = get_roc_auc(y_test, y_pred_proba)
    tpr_at_low_fpr = get_tpr_metric(y_test, y_pred_proba, fpr_budget)

    if plot_roc:
        print("ROC AUC: ",roc_auc)
        print(f'TPR@{fpr_budget}%FPR: {tpr_at_low_fpr}')
        plot_tpr_fpr_curve(y_test, y_pred_proba, fpr_budget)
        return
    else:
        return roc_auc, tpr_at_low_fpr

def hyperparam_search(X,y, dataset_name, fpr_budget):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    scores = {}
    for max_features in [10, 15, 20, 30, 32, 34, 36, 38, 40, 50, 54, 58, 62]:
    # for max_features in [10, 15]:
        for vectorizer in ['tf', 'count']:
            # print(vectorizer, "--------------------------------")
            if vectorizer == 'tf':
                converter = TfidfVectorizer(max_features=max_features)
            elif vectorizer == 'count':
                converter = CountVectorizer(max_features=max_features)
            X_train_vector = converter.fit_transform(X_train).toarray()  
            X_test_vector = converter.fit_transform(X_test).toarray()  
            X_train_df = pd.DataFrame(X_train_vector)
            X_test_df = pd.DataFrame(X_test_vector)
            models = ['multi', 'gaussian', 'random', 'gradient', 'stack']
            # models = ['multi', 'gaussian']
            for model_type in models:
                if model_type == 'random':
                    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                elif model_type == 'gaussian':
                    model = GaussianNB()
                elif model_type == 'multi':
                    model = MultinomialNB()
                elif model_type == 'gradient':
                    model = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=42)
                elif model_type == 'stack':
                    estimators = [
                    ('random', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
                    ('gaussian', GaussianNB()),
                    ('multi', MultinomialNB()),
                    ('gradient', GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=5, random_state=42))
                    ]
                    model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42))

                # print("Fiting model...", model_type)
                model.fit(X_train_df, y_train)
                y_pred = model.predict(X_test_df)
                try:
                    y_pred_proba = model.predict_proba(X_test_df)[:,1]
                except:
                    y_pred_proba = model.predict_log_proba(X_test_df)[:,1]
                roc_auc = get_roc_auc(y_test, y_pred_proba)
                print("ROC AUC: ",roc_auc)
                tpr_at_low_fpr = get_tpr_metric(y_test, y_pred_proba, fpr_budget)
                print(f'TPR@{fpr_budget}%FPR: {tpr_at_low_fpr}')

                scores[str(max_features)+"_"+vectorizer+"_"+model_type] = roc_auc

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_scores[0:10])
    best_params = sorted_scores[0][0].split('_')
    params = {}
    params['max_features'] = int(best_params[0])
    params['vectorizer'] = best_params[1]
    params['model_type'] = best_params[2]
    return params

def bag_of_words_basic(X,y, dataset_name, fpr_budget, plot_roc, hypersearch):
    default_params = {
        'wikimia': {'max_features': 34, 'vectorizer': 'tf', 'model_type': 'gaussian'},
        'bookmia': {'max_features': 58, 'vectorizer': 'count', 'model_type': 'stack'},
        'temporal_wiki': {'max_features': 52, 'vectorizer': 'tf', 'model_type': 'stack'},
        'temporal_arxiv': {'max_features': 62, 'vectorizer': 'count', 'model_type': 'stack'},
        'arxiv_tection': {'max_features': 62, 'vectorizer': 'tf', 'model_type': 'stack'},
        'book_tection': {'max_features': 54, 'vectorizer': 'tf', 'model_type': 'stack'},
        'arxiv_1m': {'max_features': None, 'vectorizer': 'tf', 'model_type': 'stack'},
        'arxiv1m_1m': {'max_features': 52, 'vectorizer': 'tf', 'model_type': 'gaussian'},
        'multi_web': {'max_features': 38, 'vectorizer': 'count', 'model_type': 'gradient'},
        'laion_mi': {'max_features': 10, 'vectorizer': 'tf', 'model_type': 'gaussian'},
        'gutenberg': {'max_features': None, 'vectorizer': 'tf', 'model_type': 'multi'},
    }

    trials = 10
    if not hypersearch: # read from defaults
        params = default_params[dataset_name]
    else: # Conduct hyperparameter search
        params = hyperparam_search(X,y, dataset_name, fpr_budget)
    print(params)

    auc_scores = []
    tpr_scores = []
    if not plot_roc: 
        for _ in range(trials):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
            roc_auc, tpr_at_low_fpr = train_classifier(X_train, X_test, y_train, y_test, fpr_budget, plot_roc, params=params)
            print(roc_auc)
            auc_scores.append(roc_auc)
            tpr_scores.append(tpr_at_low_fpr)
        
        print(f"Mean auc_score over {trials} runs: {mean(auc_scores)*100:.3f} \u00B1 {pstdev(auc_scores)*100:.0f}")
        print(f"Mean tpr@{fpr_budget}%fpr over {trials} runs: {mean(tpr_scores)*100:.3f} \u00B1 {pstdev(tpr_scores)*100:.0f}")
    else:
        # Only one run to plot the TPR vs FPR curve
        train_classifier(X_train, X_test, y_train, y_test, fpr_budget, plot_roc, params=params)




        

