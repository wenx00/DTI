import torch
import tensorflow as tf
import pandas as pd
import json
import pickle
import numpy as np
import dgl
import networkx as nx
import matplotlib.pyplot as plt


def load_csv_file(file_name):
    df = pd.read_csv(file_name, header=None)
    records = df.to_records(index=False)
    interactions_list = list(records)  # Drug_Drug_Interactions is a list of tuples
    return interactions_list
    # print(len(interactions_list))

