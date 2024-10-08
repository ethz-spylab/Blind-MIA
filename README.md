# Blind Baselines Beat Membership Inference Attacks for Foundation Models

This repository contains the code for the paper "Blind Baselines Beat Membership Inference Attacks for Foundation Models".

### Abstract

Membership inference (MI) attacks try to determine if a data sample was used to train a machine learning model. 
For foundation models trained on unknown Web data, MI attacks are often used to detect copyrighted training materials, measure test set contamination, or audit machine unlearning. 
Unfortunately, we find that evaluations of MI attacks for foundation models are flawed, because they sample members and non-members from different distributions. 
For 9 published MI evaluation datasets, we show that blind attacks—that distinguish the member and non-member distributions without looking at any trained model—outperform state-of-the-art MI attacks. 
Existing evaluations thus tell us nothing about membership leakage of a foundation model’s training data.

## How to run our attacks?
At the root of the repository, run the following to install required dependencies:

```pip install -r requirements.txt```

### Setting up Datasets

All datasets except for the Arxiv (1 month vs 1 month) dataset require no further setup, you can continue to the next step for all other datasets. (For arxiv1m_1m dataset, check [this section](#setting-up-the-arxiv-1-month-vs-1-month-dataset))

### Run Attacks

Run the ``run_attack.py`` script with the required command line arguments using the command below.

``` python3 run_attack.py --dataset <dataset> --attack <attack> ``` 

where ```<dataset>``` is one of the datasets from the following list:

```'wikimia', 'bookmia', 'temporal_wiki', 'temporal_arxiv', 'arxiv_1m', 'arxiv_1m_1m', 'multi_web', 'laion_mi', 'gutenberg' ```

and ```<attack>``` is one of the following attacks:

1. ``date_detection``: Applicable for temporal datasets ``wikimia``, ```temporal_wiki```, ```temporal_arxiv```, ``arxiv1m``, and ```arxiv1m_1m```. It infers membership based on dates extracted from the text.
2. ``bag_of_words``: Applicable for all datasets. It infers membership based on the bag-of-words representation of the text.
3. ``greedy_selection``: Applicable for all datasets but works more efficiently on datasets with shorter text samples. Gives best results on datasets: ``temporal_wiki, arxiv1m_1m, multi_web, laion_mi``

### Example:
For example, to run the bag-of-words attack on the WikiMIA dataset, run the following command:

``` python3 run_attack.py --dataset WikiMIA --attack bag_of_words ```
### Optional Flags:
To specify the FPR budget to be used to compute the TPR@x%FPR, use the ``fpr_budget`` flag and specify the desired FPR budget. For example, to compute the TPR@5%FPR, run the following command:

``` python3 run_attack.py --dataset WikiMIA --attack bag_of_words --fpr_budget 5 ```

To redo the hyper-parameter search, add the flag ``--hypersearch``, otherwise the bag of words attack uses the best default hyper-parameters. To plot the AUC ROC curve, add the flag ``--plot_roc``. 

## Results

| MI Dataset           | Metric                    | Best Attack | Ours | Blind Attack Type |
|----------------------|---------------------------|-------------|------|:-------------------:|
|                      | <span style="color:cyan"> *Temporal Shifted Datasets* </span> |             |      |                   |
| WikiMIA              | TPR@5%FPR                 |        43.2 | 94.7 | ``bag_of_words``               |
|                      | AUCROC                    |        83.9 |   99 | ``bag_of_words``               |
| BookMIA              | TPR@5%FPR                 |        33.6 | 64.5 | ``bag_of_words``               |
|                      | AUCROC                    |          88 | 91.4 | ``bag_of_words``               |
| Temporal Wiki        | TPR@1%FPR                 |             | 36.5 | ``greedy_selection``            |
|                      | AUCROC                    |        79.6 | 79.9 | ``greedy_selection``            |
| Temporal Arxiv       | TPR@1%FPR                 |             |  9.1 | ``bag_of_words``               |
|                      | AUCROC                    |        74.5 | 75.3 | ``bag_of_words``               |
| Arxiv                | TPR@1%FPR                 |         5.9 | 10.6 | ``date_detection``              |
| (all vs 1 month)     | AUCROC                    |        67.8 | 72.3 | ``date_detection``              |
| Arxiv                | TPR@1%FPR                 |         2.5 |  2.7 | ``greedy_selection``            |
| (1 month vs 1 month) |                           |             |      |                   |
|                      | <span style="color:cyan"> *Biased Replication* </span>        |             |      |                   |
| Multi-Web            | TPR@1%FPR                 |        40.3 |   93 | ``greedy_selection``            |
|                      | AUCROC                    |        81.7 |   98 | ``bag_of_words``               |
| LAION-MI             | TPR@1%FPR                 |         2.5 |  8.9 | ``greedy_selection``            |
| Gutenberg            | TPR@1%FPR                 |        18.8 | 55.1 | ``greedy_selection``            |
|                      | AUCROC                    |        85.6 | 96.1 | ``bag_of_words``               |

### Setting up the Arxiv (1 month vs 1 month) dataset

We handle this dataset separately because it is too big to push to the repository. Here are trhe steps to extract the dataset:
1. Download the whole arxiv dataset from [here]( https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/redpajama-arxiv-refine-result.jsonl).
2. Run the data extracion script which will save the processed dataset in the [arxiv1m_1m](data/arxiv1m_1m) folder. 

``` python3 data_script_1m_1m.py --path <path to the downloaded jsonl file>```

3. Run the attack on the dataset using the command below:

``` python3 run_attack.py --dataset arxiv1m_1m --attack greedy_selection ```