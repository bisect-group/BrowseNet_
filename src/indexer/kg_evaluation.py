import networkx as nx
import numpy as np
import random
import json
import argparse
import pickle as pkl
import copy
import pandas as pd
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
FILE_DIR_PARENT = FILE_DIR.parent
ROOT_DIR = FILE_DIR_PARENT.parent

def get_edge_accuracy(queries,graph):
    edge_accuracies = []
    try:
        for query in queries:
            edge_list = query['edge_list']
            if len(edge_list[0])!=0:
                edge_list_actual = [tuple(edge) for edge in edge_list]
                edge_list_pred = [edge for edge in edge_list_actual if edge in graph.edges()]
                edge_accuracy = len(edge_list_pred)/len(edge_list_actual)
                edge_accuracies.append(edge_accuracy)
        return sum(edge_accuracies)/len(edge_accuracies)
    except:
        return -1

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='musique')
    parser.add_argument('--ner_method', type=str, default='gliner') # us.anthropic.claude-3-7-sonnet-20250219-v1_0
    parser.add_argument('--similarity_method', type=str, default='colbert')
    parser.add_argument('--threshold', type=list, default=0.9)
    args = parser.parse_args()

    queries = json.load(open( ROOT_DIR / 'questions.json','r'))
    graph_keyword = pkl.load(open(ROOT_DIR / f'artifacts/{args.dataset}/graph_keyword_{args.ner_method}_colbert_{args.threshold}.pkl','rb'))
    print('Graph loaded!')
    print('Number of nodes: ',len(graph_keyword.nodes()))

    ent_dict = pkl.load(open(ROOT_DIR / f'artifacts/{args.dataset}/entities_dict_{args.ner_method}.pkl','rb'))
    print("Number of entities recognized: ",len(ent_dict['kw2chunk'].keys()))

    # getting the different graphs based on the thresholds
    graphs_edge_acc = []
    graphs_density = []
    threshold = args.threshold
    graph = copy.deepcopy(graph_keyword)
    print(f'Graph with threshold {threshold} has {len(graph.edges())} edges and {len(graph.nodes())} nodes')
    graph_edge_acc = get_edge_accuracy(queries,graph)
    graph_density = nx.density(graph)
    graphs_edge_acc.append(graph_edge_acc)
    graphs_density.append(graph_density)
    print(f'Graph with threshold {threshold} has edge accuracy {graph_edge_acc} and density {graph_density}')

    df = pd.DataFrame({'threshold':threshold,'edge_accuracy':graphs_edge_acc,'density':graphs_density})
    df.to_csv(ROOT_DIR / f'artifacts/{args.dataset}/graph_evaluation_{args.ner_method}_{args.similarity_method}_{threshold}.csv',index=False)




