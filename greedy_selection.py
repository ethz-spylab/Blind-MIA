import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
# from ortools.algorithms.python import knapsack_solver
import itertools
from collections import defaultdict
# from datasets import load_dataset

# returns the set of all N-grams, for N≤n
# e.g., if n=3 and s="ababacd", this returns
# {"a", "b", "c", "d", "ab", "ba", "ac", "cd", "aba", "bab", "bac", "acd"}
def ngrams(s, n=5):
    if n == 1:
        return set(s)
    iters = itertools.tee(s, n)                                                     
    for i, it in enumerate(iters):                                               
        next(itertools.islice(it, i, i), None)                                               
    return set("".join(x) for x in zip(*iters)).union(ngrams(s, n=n-1))

# list of sentences for each ngram
def count_unique_chars(lines, threshold=None):
    char_counts = defaultdict(set)
    for i, line in enumerate(lines):
        unique_chars = ngrams(line)
        for char in unique_chars:
            char_counts[char].add(i)
    
    if threshold is not None:
        for c in list(char_counts.keys()):
            if len(char_counts[c]) < threshold:
                del char_counts[c]

    return char_counts


def greedy_selection_basic(member_lines, nonmember_lines, dataset_name, fpr_budget, plot_roc):
    
    TEST_SIZE = 0.1
    CROSS_VALS = 10
    RES = []
    TARGET_FPR = fpr_budget

    # member_lines = list(dataset['member'])
    # nonmember_lines = list(dataset['nonmember'])
    N = int(TEST_SIZE*len(member_lines))
    # print(N)

    for _ in range(CROSS_VALS):
        member_sample = np.random.permutation(member_lines)
        nonmember_sample = np.random.permutation(nonmember_lines)

        # make a train-test split
        member_test = member_sample[-N:]
        nonmember_test = nonmember_sample[-N:]

        member_sample = member_sample[:-N]
        nonmember_sample = nonmember_sample[:-N]

        # only consider ngrams that appear at least THRESHOLD times in the
        # training set to guard against overfitting
        THRESHOLD = 1

        # Count ngrams in member and nonmember captions (train)
        member_char_counts = count_unique_chars(member_sample, threshold=THRESHOLD)
        nonmember_char_counts = count_unique_chars(nonmember_sample)

        # Count ngrams in member and nonmember captions (test)
        member_char_counts_test = count_unique_chars(member_test)
        nonmember_char_counts_test = count_unique_chars(nonmember_test)

        BUDGET = int(1.0 * len(nonmember_sample))
        CURR_TPR = 0
        CURR_FPR = 0
        CURR_TPR_TEST = 0
        CURR_FPR_TEST = 0
        sol_chars = []

        best_tpr = 0

        while CURR_FPR < BUDGET:
            candidates = [c for c in sorted(member_char_counts.keys()) if len(nonmember_char_counts[c]) <= BUDGET - CURR_FPR]
            if len(candidates) == 0:
                break
            ratios = [(len(member_char_counts[c]) + 0) / (len(nonmember_char_counts[c])+1) for c in candidates]
            best_idx = np.argmax(ratios)
            chosen_c = candidates[best_idx]
            sol_chars.append(chosen_c)

            CURR_TPR += len(member_char_counts[chosen_c])
            CURR_FPR += len(nonmember_char_counts[chosen_c])

            #print("TPR", sorted(list(member_char_counts[chosen_c])))
            #print("FPR", sorted(list(nonmember_char_counts[chosen_c])))

            temp = member_char_counts[chosen_c].copy()
            for c in list(member_char_counts.keys()):
                member_char_counts[c] -= temp
                if len(member_char_counts[c]) == 0:
                    del member_char_counts[c]

            temp = nonmember_char_counts[chosen_c].copy()
            for c in list(nonmember_char_counts.keys()):
                nonmember_char_counts[c] -= temp
                if len(nonmember_char_counts[c]) == 0:
                    del nonmember_char_counts[c]

            CURR_TPR_TEST += len(member_char_counts_test[chosen_c])
            CURR_FPR_TEST += len(nonmember_char_counts_test[chosen_c])

            temp = member_char_counts_test[chosen_c].copy()
            for c in list(member_char_counts_test.keys()):
                member_char_counts_test[c] -= temp
                if len(member_char_counts_test[c]) == 0:
                    del member_char_counts_test[c]

            temp = nonmember_char_counts_test[chosen_c].copy()
            for c in list(nonmember_char_counts_test.keys()):
                nonmember_char_counts_test[c] -= temp
                if len(nonmember_char_counts_test[c]) == 0:
                    del nonmember_char_counts_test[c]

            #print(chosen_c, 
            #      100 * CURR_TPR / len(member_sample), 
            #      100 * CURR_FPR / len(nonmember_sample), 
            #      100 * CURR_TPR_TEST / len(member_test), 
            #      100 * CURR_FPR_TEST / len(nonmember_test))

            m = [i for i,s in enumerate(member_sample) if len(ngrams(s).intersection(sol_chars)) > 0]
            nm = [s for i,s in enumerate(nonmember_sample) if len(ngrams(s).intersection(sol_chars)) > 0]
            TPR_tr = 100 * len(m) / len(member_sample)
            FPR_tr = 100 * len(nm) / len(nonmember_sample)
            TPR = 100 * len([s for s in member_test if len(ngrams(s).intersection(sol_chars)) > 0]) / len(member_test)
            FPR = 100 * len([s for s in nonmember_test if len(ngrams(s).intersection(sol_chars)) > 0]) / len(nonmember_test)

            print(f"train TPR = {TPR_tr:.2f} @ {FPR_tr:.2f} FPR", f"test TPR = {TPR:.2f} @ {FPR:.2f} FPR")
            
            if FPR > TARGET_FPR:
                RES.append(best_tpr)
                print("BEST", best_tpr)
                break
            else:
                best_tpr = max(best_tpr, TPR)

    print(RES)
    if len(RES) == 0:
        print(f'TPR@{fpr_budget}%FPR over {CROSS_VALS} runs: 0.0')
    else:
        print(f'TPR@{fpr_budget}%FPR over {CROSS_VALS} runs: {np.mean(RES)}')


def greedy_selection_wiki(member_lines, nonmember_lines, dataset_name, fpr_budget, plot_roc):
    random.seed(42)  # For reproducibility
    CROSS_VALS = 10
    TARGET_FPR = 1
    ALL_AUCS = []

    for _ in range(CROSS_VALS):
        FPRS = []
        TPRS = []
        member_sample = np.random.permutation(member_lines)
        nonmember_sample = np.random.permutation(nonmember_lines)

        # make a train-test split
        N = 20
        member_test = member_sample[-N:]
        nonmember_test = nonmember_sample[-N:]

        member_sample = member_sample[:-N]
        nonmember_sample = nonmember_sample[:-N]

        # only consider ngrams that appear at least THRESHOLD times in the
        # training set to guard against overfitting
        THRESHOLD = 1

        # Count ngrams in member and nonmember captions (train)
        member_char_counts = count_unique_chars(member_sample, threshold=THRESHOLD)
        nonmember_char_counts = count_unique_chars(nonmember_sample)

        # Count ngrams in member and nonmember captions (test)
        member_char_counts_test = count_unique_chars(member_test)
        nonmember_char_counts_test = count_unique_chars(nonmember_test)

        BUDGET = int(1.0 * len(nonmember_sample))
        CURR_TPR = 0
        CURR_FPR = 0
        CURR_TPR_TEST = 0
        CURR_FPR_TEST = 0
        sol_chars = []

        best_tpr = 0

        while CURR_FPR < BUDGET:
            candidates = [c for c in member_char_counts.keys() if len(nonmember_char_counts[c]) <= BUDGET - CURR_FPR]
            if len(candidates) == 0:
                break
            ratios = [(len(member_char_counts[c]) - 5) / (len(nonmember_char_counts[c]) + 5) for c in candidates]
            best_idx = np.argmax(ratios)
            chosen_c = candidates[best_idx]
            sol_chars.append(chosen_c)

            CURR_TPR += len(member_char_counts[chosen_c])
            CURR_FPR += len(nonmember_char_counts[chosen_c])

            #print("TPR", sorted(list(member_char_counts[chosen_c])))
            #print("FPR", sorted(list(nonmember_char_counts[chosen_c])))

            temp = member_char_counts[chosen_c].copy()
            for c in list(member_char_counts.keys()):
                member_char_counts[c] -= temp
                if len(member_char_counts[c]) == 0:
                    del member_char_counts[c]

            temp = nonmember_char_counts[chosen_c].copy()
            for c in list(nonmember_char_counts.keys()):
                nonmember_char_counts[c] -= temp
                if len(nonmember_char_counts[c]) == 0:
                    del nonmember_char_counts[c]

            CURR_TPR_TEST += len(member_char_counts_test[chosen_c])
            CURR_FPR_TEST += len(nonmember_char_counts_test[chosen_c])

            temp = member_char_counts_test[chosen_c].copy()
            for c in list(member_char_counts_test.keys()):
                member_char_counts_test[c] -= temp
                if len(member_char_counts_test[c]) == 0:
                    del member_char_counts_test[c]

            temp = nonmember_char_counts_test[chosen_c].copy()
            for c in list(nonmember_char_counts_test.keys()):
                nonmember_char_counts_test[c] -= temp
                if len(nonmember_char_counts_test[c]) == 0:
                    del nonmember_char_counts_test[c]


            m = [i for i,s in enumerate(member_sample) if len(ngrams(s, n=10).intersection(sol_chars)) > 0]
            nm = [s for i,s in enumerate(nonmember_sample) if len(ngrams(s, n=10).intersection(sol_chars)) > 0]
            TPR_tr = 100 * len(m) / len(member_sample)
            FPR_tr = 100 * len(nm) / len(nonmember_sample)
            TPR = 100 * len([s for s in member_test if len(ngrams(s, n=10).intersection(sol_chars)) > 0]) / len(member_test)
            FPR = 100 * len([s for s in nonmember_test if len(ngrams(s, n=10).intersection(sol_chars)) > 0]) / len(nonmember_test)

            print(f"train TPR = {TPR_tr:.2f} @ {FPR_tr:.2f} FPR", f"test TPR = {TPR:.2f} @ {FPR:.2f} FPR")
            FPRS.append(FPR)
            TPRS.append(TPR)

        print(FPRS)
        print(TPRS)

        FPRS.append(100.0)
        TPRS.append(100.0)

        FPRS = np.asarray(FPRS) / 100.0
        TPRS = np.asarray(TPRS) / 100.0

        idx = np.argsort(FPRS)
        FPRS = FPRS[idx]
        TPRS = np.maximum.accumulate(TPRS[idx])

        MAX_AUC = 0
        for i in range(2, len(FPRS)):
            AUC = auc(list(FPRS[:i]) + [1.0], list(TPRS[:i]) + [1.0])
            MAX_AUC = max(MAX_AUC, AUC)
        print(MAX_AUC)
        ALL_AUCS.append(MAX_AUC)

    print(np.mean(ALL_AUCS))
        
def test(w, members_test, nonmembers_test):
	a = 100 * len([m for m in nonmembers_test if sum([wi in m for wi in w]) > 0]) / len(nonmembers_test)
	b = 100 * len([m for m in members_test if sum([wi in m for wi in w]) > 0]) / len(members_test)
	return a,b

def greedy_selection_arxiv(members, nonmembers, dataset_name, fpr_budget, plot_roc):
    N = 2000
    members = [''.join(e for e in m if e.isalnum() or e.isspace()) for m in members]
    nonmembers = [''.join(e for e in m if e.isalnum() or e.isspace()) for m in nonmembers]
    members_test = members[:N]
    nonmembers_test = nonmembers[:N]
    members = members[N:]
    nonmembers = nonmembers[N:]

    best_words_ref = ["non-normality", "SED-fitting", "enactment", "classification-regression", "online-to-batch", "far-off", "single-turn",  "GWAS", "lemmatized", "Relic", "articulatory",  "Serra", "Ashtekar", "state-actions", "FDD", "UVW", "Nf3", "substation", "second-price", "unfoldings", "per-node", "Supervisory", "pushback", "MMoE"]
    print(test(best_words_ref, members_test, nonmembers_test))

    best_words = ["pre-registered", "speaker-dependent", "MMoE", "pushback", "Supervisory", "per-node", "fAB", "unfoldings", "second-price", "substation", "Nf3", "UVW", "FDD", "state-actions", "Ashtekar", "Serra"]
    print(test(best_words, members_test, nonmembers_test))

    m_dict = {}
    for m in members:
        words = list(set(m.split(" ")))
        for w in words:
            if w in m_dict:
                m_dict[w] += 1
            else:
                m_dict[w] = 1
                
    nm_dict = {}
    for m in nonmembers:
        words = list(set(m.split(" ")))
        for w in words:
            if w in nm_dict:
                nm_dict[w] += 1
            else:
                nm_dict[w] = 1
                
    mtest_dict = {}
    for m in members_test:
        words = list(set(m.split(" ")))
        for w in words:
            if w in mtest_dict:
                mtest_dict[w] += 1
            else:
                mtest_dict[w] = 1
                
    nmtest_dict = {}
    for m in nonmembers_test:
        words = list(set(m.split(" ")))
        for w in words:
            if w in nmtest_dict:
                nmtest_dict[w] += 1
            else:
                nmtest_dict[w] = 1

    words = list(m_dict.keys())
    tprs = [m_dict[w] for w in words]
    fprs = [1 if w not in nm_dict else nm_dict[w]+1 for w in words]
    tprs2 = [tprs[i] if tprs[i] > 10 else 0 for i in range(len(words))]
    best = np.argsort(np.asarray(tprs2) / np.asarray(fprs))[::-1][:100]

    best_w = []
    for i in range(100):
        w = words[best[i]]
        if sum([e.isspace() for e in w]) == 0:
            best_w.append(w)
            print(100*fprs[best[i]] / len(nonmembers), 100*tprs[best[i]] / len(members), w, test([w], members_test, nonmembers_test))
        else:
            print(f"skipping {w}")

    print("="*80)
    print("="*80)
    print("="*80)
    print("="*80)

    words = list(nm_dict.keys())
    tnrs = [nm_dict[w] for w in words]
    fnrs = [1 if w not in m_dict else m_dict[w]+1 for w in words]
    tnrs2 = [tnrs[i] if tnrs[i] > 10 else 0 for i in range(len(words))]
    best = np.argsort(np.asarray(tnrs2) / np.asarray(fnrs))[::-1][:200]

    best_w = []
    for i in range(100):
        w = words[best[i]]
        if sum([e.isspace() for e in w]) == 0:
            best_w.append(w)
            print(100*fnrs[best[i]] / len(members), 100*tnrs[best[i]] / len(nonmembers), w, test([w], members_test, nonmembers_test))
        else:
            print(f"skipping {w}")