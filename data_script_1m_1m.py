import ast
import json
import numpy as np
import argparse

# Opening JSON file

parser = argparse.ArgumentParser(description='Extraction script for arxiv 1 month vs 1 month dataset')
parser.add_argument('--path', help='input jsonl file path', type=str, required=True)
args = parser.parse_args()
print(args.path)

f = open(args.path, 'r')
data = f.readlines()

members = []
nonmembers = []

for d in data:
	temp = ast.literal_eval(d)

	text = temp["text"]
	date = temp["meta"]["yymm"]
	year = int(date[:2])
	month = int(date[2:])
	if year <= 22 or year > 50:
		continue
	else:
		assert year == 23, (date, year)
		if month == 2:
			members.append(text)
		elif month == 3:
			nonmembers.append(text)

print(len(members))
print(len(nonmembers))
np.save("data/arxiv1m_1m/member.npy", np.asarray(members))
np.save("data/arxiv1m_1m/nonmember.npy", np.asarray(nonmembers))