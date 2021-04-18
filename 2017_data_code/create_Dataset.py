from utils import *
import json
import pickle


# -------------Create DDI dataset-------------
DDI_list = load_csv_file('DDI_origin.csv')
DDI_dataset = []
Drug_dict = {}
for index, DDI_pair in enumerate(DDI_list):
    if DDI_pair[0] not in Drug_dict:
        Drug_dict[DDI_pair[0]] = len(Drug_dict)
    if DDI_pair[1] not in Drug_dict:
        Drug_dict[DDI_pair[1]] = len(Drug_dict)

    DDI_dataset.append((Drug_dict[DDI_pair[0]], Drug_dict[DDI_pair[1]]))

with open('DDI_dataset.csv', 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in DDI_dataset))

# -------------Create DPI dataset-------------
DPI_list = load_csv_file('DPI_origin.csv')
DPI_dataset = []
Protein_dict = {}
for index, DPI_pair in enumerate(DPI_list):
    if DPI_pair[0] not in Drug_dict:
        Drug_dict[DPI_pair[0]] = len(Drug_dict)
    if DPI_pair[1] not in Protein_dict:
        Protein_dict[DPI_pair[1]] = len(Protein_dict)

    DPI_dataset.append((Drug_dict[DPI_pair[0]], Protein_dict[DPI_pair[1]]))

with open('DPI_dataset.csv', 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in DPI_dataset))


# -------------Create PDI dataset-------------
# PDI_list = load_csv_file('DPI_dataset.csv')
# PDI_dataset =[]
# for i, val in enumerate(PDI_list):
#     PDI_dataset.append((val[1], val[0]))
# with open('PDI_dataset.csv', 'w') as f:
#     f.write('\n'.join('%s,%s' % x for x in PDI_dataset))


# -------------Create PPI dataset-------------
PPI_list = load_csv_file('PPI_origin.csv')
PPI_dataset = []
for index, PPI_pair in enumerate(PPI_list):
    if PPI_pair[0] not in Protein_dict:
        Protein_dict[PPI_pair[0]] = len(Protein_dict)
    if PPI_pair[1] not in Protein_dict:
        Protein_dict[PPI_pair[1]] = len(Protein_dict)

    PPI_dataset.append((Protein_dict[PPI_pair[0]], Protein_dict[PPI_pair[1]]))

with open('PPI_dataset.csv', 'w') as fp:
    fp.write('\n'.join('%s,%s' % x for x in PPI_dataset))


with open('Drug_dict.json', 'w') as fp:
    json.dump(Drug_dict, fp)
with open('Protein_dict.json', 'w') as fp:
    json.dump(Protein_dict, fp)

# # JSON load:
# with open('data.json', 'r') as fp:
#     data = json.load(fp)


# with open('Drug_dict.pkl', 'wb') as f:
#     pickle.dump(Drug_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open('Protein_dict.pkl', 'wb') as f:
#     pickle.dump(Protein_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open('DDI_dict.pkl', 'wb') as f:
#     pickle.dump(Drug_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open('parrot.pkl', 'rb') as f:
#     mynewlist = pickle.load(f)
