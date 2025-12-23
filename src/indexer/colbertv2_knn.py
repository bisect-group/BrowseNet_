#### This file is modifed from the original colbertv2_knn.py file in the HippoRAG repository ####
import os
import pandas as pd
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
import pickle as pkl
import regex as re
from pathlib import Path
import copy 
from ..helpers.utils import preprocess_entity as processing_phrases


FILE_DIR = Path(__file__).resolve().parent
FILE_DIR_PARENT = FILE_DIR.parent
ROOT_DIR = FILE_DIR_PARENT.parent

def make_entity_csv(dataset, ner_method):
    entities_dict_path = ROOT_DIR / f'artifacts/{dataset}/entities_dict_{ner_method}.pkl'
    entities_dict = pkl.load(open(entities_dict_path, 'rb'))
    kws_set = set()
    for _,kws in entities_dict['chunk2kw'].items():
        for kw in kws:
            kws_set.add(kw)
    
    unique_phrases = list(kws_set)
    kb = pd.DataFrame(unique_phrases, columns=['strings'])
    kb2 = copy.deepcopy(kb)
    kb['type'] = 'query'
    kb2['type'] = 'kb'
    kb_full = pd.concat([kb, kb2])
    kb_full_path = ROOT_DIR / f'artifacts/{dataset}/kb_to_kb_{ner_method}.tsv'
    kb_full.to_csv(kb_full_path, sep='\t')
    return 

def retrieve_knn(kb, queries, duplicate=True, nns=100):
    checkpoint_path = str((FILE_DIR / 'exp' / 'colbertv2.0').resolve())

    if duplicate:
        kb = list(set(list(kb) + list(queries)))

    colbert_dir = str((FILE_DIR).resolve())
    corpus_tsv_path = str((FILE_DIR / 'colbert' / 'corpus.tsv').resolve())
    with open(corpus_tsv_path, 'w') as f:
        for pid, p in enumerate(kb):
            f.write(f"{pid}\t\"{p}\"" + '\n')
    queries_tsv_path = str((FILE_DIR / 'colbert' / 'queries.tsv').resolve())
    with open(queries_tsv_path, 'w') as f:
        for qid, q in enumerate(queries):
            f.write(f"{qid}\t{q}" + '\n')

    with Run().context(RunConfig(nranks=1, experiment="colbert", root=colbert_dir)):
        config = ColBERTConfig(
            nbits=2,
            root=str(FILE_DIR / 'colbert')
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name="nbits_2", collection=corpus_tsv_path, overwrite=True)

    with Run().context(RunConfig(nranks=1, experiment="colbert", root=colbert_dir)):
        config = ColBERTConfig(
            root=str(FILE_DIR / 'colbert')
        )
        searcher = Searcher(index="nbits_2", config=config)
        queries_obj = Queries(queries_tsv_path)
        ranking = searcher.search_all(queries_obj, k=nns)

    ranking_dict = {}

    for i in range(len(queries)):
        query = queries[i]
        rank = ranking.data[i]
        max_score = rank[0][2]
        if duplicate:
            rank = rank[1:]
        ranking_dict[query] = ([kb[r[0]] for r in rank], [r[2] / max_score for r in rank])

    return ranking_dict

def get_nearest_neighbors(dataset, ner_method):
    output_path = ROOT_DIR / 'artifacts' / dataset / f'nearest_neighbor_{ner_method}.pkl'
    if os.path.exists(output_path):
        print('Nearest neighbors already exist!')
        return
    string_filename = ROOT_DIR / "artifacts" / dataset / f"kb_to_kb_{ner_method}.tsv"
    string_df = pd.read_csv(string_filename, sep='\t')
    string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]

    queries = string_df[string_df.type == 'query']
    kb = string_df[string_df.type == 'kb']

    nearest_neighbors = retrieve_knn(kb.strings.values, queries.strings.values)
    pkl.dump(nearest_neighbors, open(output_path, 'wb'))
    print('Saved nearest neighbors to {}'.format(output_path))
    return 