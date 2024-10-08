import regex as re
from sklearn.metrics import classification_report
from utils import *

def date_detection_basic(X,y, dataset_name, fpr_budget, plot_roc):
    cutoff_year=2023
    if 'wikimia' == dataset_name:
        cutoff_year=2023
    if 'temporal_wiki' == dataset_name:
        cutoff_year=2021
    if 'temporal_arxiv' == dataset_name:
        cutoff_year=2020

    y_true = y
    y_pred = [0]*len(y)
    y_pred_proba = [0]*len(y)

    incorrect_count = 0
    no_year_count = 0
    for index in range(len(X)):
        x = re.findall("[0-9][0-9][0-9][0-9]", X[index])
        if len(x)>0:
            year = max([int(y) for y in x])
            if year < cutoff_year:
                y_pred[index] = 1
                y_pred_proba[index] = 1.0
            else:
                y_pred[index] = 0
                y_pred_proba[index] = 0.0
        else:             # "NO YEAR"
            no_year_count +=1
            y_pred[index] = 0
            y_pred_proba[index] = 0.5

    print("- no of datapoints with missing years:", no_year_count)
    print(classification_report(y_true, y_pred, target_names=['non-member', 'member']))

    roc_auc = get_roc_auc(y_true, y_pred_proba)
    print("ROC AUC: ",roc_auc)
    tpr_at_low_fpr = get_tpr_metric(y_true, y_pred_proba, fpr_budget)
    print(f'TPR@{fpr_budget}%FPR: {tpr_at_low_fpr}')

    if plot_roc:
        plot_tpr_fpr_curve(y_true, y_pred_proba, fpr_budget)

def is_year(s):
    if len(s) == 2:
        return (0 <= int(s) <= 23) or (70 <= int(s) <= 99)
    elif len(s) == 4:
        return 1970 <= int(s) <= 2023
    else:
        return False

def to_year(d):
    if d <= 23:
        return 2000 + d
    elif d <= 99:
        return 1900 + d
    else:
        assert d >= 1970, d
        return d

def max_year(s):
    pattern = re.compile(r'(?:\\cite[p|t]?\{|(?<!^)\G),?\s*([^,}]+)+')
    matches = re.findall(pattern, s)
    # print(matches)
    numbers = [re.findall(r'\d+', m[0]) for m in matches]
    numbers = [n[0] if len(n) == 1 else "9999" for n in numbers]
    numbers = [int(n) if is_year(n) else 9999 for n in numbers]
    years = [to_year(n) for n in numbers]
    
    if len(years) > 0:
        return max(years)
    else:
        return 9999

def date_detection_arxiv(X, y, members, nonmembers, dataset_name, fpr_budget, plot_roc):
    cutoff_year=2022
    y_true = y
    y_pred = [0]*len(y)
    y_pred_proba = [0]*len(y)

    nonmember_years = []
    for m in nonmembers:
        y = max_year(m)
        nonmember_years.append(y)

    nonmember_years = np.asarray(nonmember_years)
    # np.save("arxiv_nonmembers_years_l.npy", nonmember_years)

    member_years = []
    for m in members:
        y = max_year(m)
        member_years.append(y)

    member_years = np.asarray(member_years)
    # np.save("arxiv_members_years_l.npy", member_years)
    tprs = []
    fprs = []
    for t in range(1990, 2023):
        FPR = 100 * np.mean(nonmember_years < t)
        fprs.append(FPR)
        TPR = 100 * np.mean(member_years < t)
        tprs.append(TPR)
        # print(f"{t}, TPR={TPR:.2f}, FPR={FPR:.2f}")

    for i in range(len(tprs)-1,-1,-1):
        if fprs[i] <= fpr_budget:
            print(f'TPR@{fpr_budget}%FPR: {tprs[i]:.2f}')
            break



    for index in range(len(X)):
        citations = re.findall("cite{.+}", X[index])
        # print(len(citations), citations)
        dates = []
        for cite in citations:
            dates = dates + re.findall("[0-9][0-9][0-9][0-9]", cite)
        # print(dates)
 
        if len(dates)>0:
            years = [int(y) for y in dates]
            year = max(years)

            if year < cutoff_year:
                y_pred[index] = 1
                y_pred_proba[index] = 1.0
            else:
                y_pred[index] = 0
                y_pred_proba[index] = 0.0
        else:             # "NO YEAR"

            y_pred[index] = 0
            y_pred_proba[index] = 0.5

    # print("- no of datapoints with missing years:", no_year_count)
    print(classification_report(y_true, y_pred, target_names=['non-member', 'member']))

    roc_auc = get_roc_auc(y_true, y_pred_proba)
    print("ROC AUC: ",roc_auc)
    tpr_at_low_fpr = get_tpr_metric(y_true, y_pred_proba, fpr_budget)
    print(f'TPR@{fpr_budget}%FPR: {tpr_at_low_fpr}')

    if plot_roc:
        plot_tpr_fpr_curve(y_true, y_pred_proba, fpr_budget)








