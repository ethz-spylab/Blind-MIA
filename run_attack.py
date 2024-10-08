import argparse
from date_detection import *
from bag_of_words import *
from greedy_selection import *
from utils import *

np.random.seed(0) # For reproducibility

parser = argparse.ArgumentParser(description='Blind Attacks on Membership Inference Attack Evaluation Datasets')
parser.add_argument('--dataset', help='dataset name', choices=['wikimia', 'bookmia', 'temporal_wiki', 'temporal_arxiv', 'arxiv_tection', 'book_tection', 'arxiv_1m', 'arxiv1m_1m', 'multi_web', 'laion_mi', 'gutenberg'], default='bookmia')
parser.add_argument('--attack', help='attack method', choices=['date_detection','bag_of_words','greedy_selection'], default='bag_of_words')
parser.add_argument('--plot_roc', help='set to plot FPR vs TPR curve', action="store_true")
parser.add_argument('--hypersearch', help='set to redo hyperparam search instead of using default params', action="store_true")
parser.add_argument('--fpr_budget', help='x for computing TPR@x%FPR', type=float, default=1)

args = parser.parse_args()
print(args.dataset, args.attack)

X, y, members, nonmembers = get_dataset(args.dataset)
dataset_name = args.dataset

if args.attack == 'date_detection':
    if dataset_name == 'arxiv_1m':
        date_detection_arxiv(X, y, members, nonmembers, dataset_name=dataset_name, fpr_budget=args.fpr_budget, plot_roc=args.plot_roc)
    else:
        date_detection_basic(X,y, dataset_name=dataset_name, fpr_budget=args.fpr_budget, plot_roc=args.plot_roc)
elif args.attack == 'bag_of_words':
    bag_of_words_basic(X,y, dataset_name=dataset_name, fpr_budget=args.fpr_budget, plot_roc=args.plot_roc, hypersearch=args.hypersearch)
elif args.attack == 'greedy_selection':
    if dataset_name == 'temporal_wiki':
        greedy_selection_wiki(members, nonmembers, dataset_name, args.fpr_budget, args.plot_roc)
    elif dataset_name == 'arxiv1m_1m':
        greedy_selection_arxiv(members, nonmembers, dataset_name, args.fpr_budget, args.plot_roc)
    else:
        greedy_selection_basic(members, nonmembers, dataset_name, args.fpr_budget, args.plot_roc)
