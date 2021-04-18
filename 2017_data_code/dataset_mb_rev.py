from ..utils import *
import json

DDI_list = load_csv_file('DDI_origin.csv')
DPI_list = load_csv_file('DPI_origin.csv')
PPI_list = load_csv_file('PPI_origin.csv')

dataset = []
dataset_dict = {}

for index, DDI_pair in enumerate(DDI_list):
    if DDI_pair[0] not in dataset_dict:
        dataset_dict[DDI_pair[0]] = len(dataset_dict)
    if DDI_pair[1] not in dataset_dict:
        dataset_dict[DDI_pair[1]] = len(dataset_dict)

    dataset.append((dataset_dict[DDI_pair[0]], 0, dataset_dict[DDI_pair[1]]))

for index, DPI_pair in enumerate(DPI_list):
    if DPI_pair[0] not in dataset_dict:
        dataset_dict[DPI_pair[0]] = len(dataset_dict)
    if DPI_pair[1] not in dataset_dict:
        dataset_dict[DPI_pair[1]] = len(dataset_dict)

    dataset.append((dataset_dict[DPI_pair[0]], 1, dataset_dict[DPI_pair[1]]))

for index, PPI_pair in enumerate(PPI_list):
    if PPI_pair[0] not in dataset_dict:
        dataset_dict[PPI_pair[0]] = len(dataset_dict)
    if PPI_pair[1] not in dataset_dict:
        dataset_dict[PPI_pair[1]] = len(dataset_dict)

    dataset.append((dataset_dict[PPI_pair[0]], 2, dataset_dict[PPI_pair[1]]))

with open('Hetero_dataset.csv', 'w') as fp:
    fp.write('\n'.join('%s,%s,%s' % x for x in dataset))

with open('dataset_dict.json', 'w') as f:
    json.dump(dataset_dict, f)

print('OK')
