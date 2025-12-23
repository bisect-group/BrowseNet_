import os
from collections import defaultdict
from tqdm import tqdm
import pickle as pkl
import regex as re
import copy 
import networkx as nx
import pathlib
from datetime import datetime
from dateutil.parser import parse

FILE_DIR = pathlib.Path(__file__).resolve().parent
FILE_DIR_PARENT = FILE_DIR.parent
ROOT_DIR = FILE_DIR_PARENT.parent

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def is_date(string):
    try:
        parse(string, fuzzy=False)
        return True
    except ValueError:
        return False


def get_colbert_similar_keywords(dataset, ner_method, threshold):
    
    new_entities_dict_path = ROOT_DIR / f'artifacts/{dataset}/new_entities_dict_{ner_method}_colbert_{threshold}.pkl'
    if os.path.exists(new_entities_dict_path):
        print("Similar entities dict already exists!")
        return
    
    entities_dict_path = ROOT_DIR / f'artifacts/{dataset}/entities_dict_{ner_method}.pkl'
    entities_dict = pkl.load(open(entities_dict_path, 'rb'))

    kws_set = set()
    for _,kws in entities_dict['chunk2kw'].items():
        for kw in kws:
            kws_set.add(kw)
    
    unique_phrases = list(kws_set)

    similar_ents_path = ROOT_DIR / f'artifacts/{dataset}/nearest_neighbor_{ner_method}.pkl'

    if not os.path.exists(similar_ents_path):
        print("Error occured, couldn't extract similar entities.", "Try to run colbertv2_knn.nearest_neighbors() first!")

    similar_ents = pkl.load(open(similar_ents_path, 'rb'))
    
    k2simk = defaultdict(set)
    for kw,syns in similar_ents.items():
        if kw != 'nan' and is_float(kw) == False:
            if is_date(kw) ==False:
                for syn,score in zip(*syns):
                    if syn != 'nan' and score > threshold:
                        k2simk[kw].add(syn)
                        k2simk[syn].add(kw)
    for key in k2simk:
        k2simk[key] = list(k2simk[key])

    new_entities_dict = copy.deepcopy(entities_dict)
    for chunk in entities_dict['chunk2kw'].keys():
        for kw in entities_dict['chunk2kw'][chunk]:
            for syn in k2simk[kw]:
                new_entities_dict['chunk2kw'][chunk].append(syn)

    for key in new_entities_dict['chunk2kw']:
        new_entities_dict['chunk2kw'][key] = list(set(new_entities_dict['chunk2kw'][key]))

    kw2chunk = {k:[] for k in unique_phrases}
    for chunk in new_entities_dict['chunk2kw'].keys():
        for kw in new_entities_dict['chunk2kw'][chunk]:
            try:
                kw2chunk[kw].append(chunk)
            except:
                continue
    for key in kw2chunk:
        kw2chunk[key] = list(set(kw2chunk[key]))
    new_entities_dict = {'kw2chunk':kw2chunk,'chunk2kw':new_entities_dict['chunk2kw'],'all_kws':unique_phrases,'id2chunk':entities_dict['id2chunk'],'k2simk':k2simk}
    pkl.dump(new_entities_dict, open(new_entities_dict_path, 'wb'))
    return

def construct_graph(dataset, chunk_dict, ner_method, threshold):
    graph_path = ROOT_DIR / f'artifacts/{dataset}/graph_keyword_{ner_method}_colbert_{threshold}.pkl'
    if os.path.exists(graph_path):
        print("Graph already exists!")
        return
    
    new_entities_dict_path = ROOT_DIR / f'artifacts/{dataset}/new_entities_dict_{ner_method}_colbert_{threshold}.pkl'
    new_entities_dict = pkl.load(open(new_entities_dict_path, 'rb'))
    
    new_entities_dicts = [new_entities_dict]
   
    G = nx.MultiGraph()  
    print("Creating edges...\n")
    
    for chunk in new_entities_dict['chunk2kw'].keys():
        G.add_node(chunk, chunk=new_entities_dict['id2chunk'][chunk], title=chunk_dict[chunk]['title'])

    G_keyword = G.copy()

    for i in tqdm(range(len(new_entities_dict['chunk2kw'].keys())),total=len(new_entities_dict['chunk2kw'].keys())):
        kws1 = [new_entities_dict['chunk2kw'][i] for new_entities_dict in new_entities_dicts]
        for j in range(i + 1, len(new_entities_dict['chunk2kw'].keys())):
            kws2 = [new_entities_dict['chunk2kw'][j] for new_entities_dict in new_entities_dicts]
            for kw_ in range(len(kws1)):
                if len(set(kws1[kw_]).union(kws2[kw_])) == 0:
                    entity_score = 0
                else:
                    entity_score = len(set(kws1[kw_]).intersection(kws2[kw_])) / len(set(kws1[kw_]).union(kws2[kw_]))
                if(entity_score>0):
                    G_keyword.add_edge(i, j, weight=entity_score, threshold=threshold)

    pkl.dump(G_keyword, open(graph_path, 'wb'))

    return
