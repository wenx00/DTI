# from utils import *
import json
import pickle
import pandas as pd


def load_csv_file(file_name):
    df = pd.read_csv(file_name, header=None)
    records = df.to_records(index=False)
    interactions_list = list(records)  # Drug_Drug_Interactions is a list of tuples
    return interactions_list


dataset = []

# dict. key is name, value is number.
Drug_dict = {}
Protein_dict = {}

# -------------Create 2017_DTI dataset-------------
DPI_list = load_csv_file('2017.csv')
DPI_dataset = []
for index, DPI_pair in enumerate(DPI_list):
    if DPI_pair[0] not in Drug_dict:
        Drug_dict[DPI_pair[0]] = len(Drug_dict)
    if DPI_pair[1] not in Protein_dict:
        Protein_dict[DPI_pair[1]] = len(Protein_dict)

    DPI_dataset.append((Drug_dict[DPI_pair[0]], Protein_dict[DPI_pair[1]]))

with open('2017_dataset_processed.csv', 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in DPI_dataset))
with open('Drug_dict.json', 'w') as f:
    json.dump(Drug_dict, f)
with open('Protein_dict.json', 'w') as f:
    json.dump(Protein_dict, f)
with open('2017_DPI_dataset.pkl', 'wb') as f:
    pickle.dump(DPI_dataset, f)
