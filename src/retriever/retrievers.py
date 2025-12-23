from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import networkx as nx
from tqdm import tqdm
from collections.abc import Iterable
from itertools import product
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def get_index(l,elem,max_len):
    if elem in l:
        return l.index(elem)
    else:
        return max_len

def split_qs(input_string):
    qa_pairs = re.split(r"Q\d+\)", input_string)
    qa_pairs = qa_pairs[1:]
    cleaned_qa_pairs = []
    for qa_pair in qa_pairs:
      cleaned_qa_pair = qa_pair.strip()
      cleaned_qa_pair = re.sub(r"\s+", " ", cleaned_qa_pair) 
      cleaned_qa_pairs.append(cleaned_qa_pair)
    return cleaned_qa_pairs

def split_q_tags(text):
    match = re.match(r'<(Q\d+(?:, Q\d+)*)>', text)
    if match:
        numbers = match.group(1).split(', ')
        return ' '.join(f'<{num}>' for num in numbers) + text[match.end():]
    return text  

def parse_input(input_string):
    input_string = split_q_tags(input_string)
    matches = re.findall(r'<Q(\d+)>', input_string)
    question = re.sub(r'<Q\d+>\s*', '', input_string).strip()
    return [tuple(map(int, matches)), question]

def get_q_subgraph(q):
    G = nx.DiGraph()
    split_q = split_qs(q)
    parsed_q = {i+1:parse_input(q) for i,q in enumerate(split_q)}
    for k in parsed_q.keys():
        G.add_node(k, question=parsed_q[k][1])
        if len(parsed_q[k][0])>0:
            for i in parsed_q[k][0]:
                G.add_edge(i,k)
    return G

def flatten(iterable):
    result = []
    for item in iterable:
        if isinstance(item, (list, tuple, set)):
            result.extend(flatten(item))  
        else:
            result.append(item)
    return result

def update_score_pred_dict(node, combo, top_k, cos_sims, score_pred_dict, preds, depth):
    for id,k in enumerate(top_k):
        if k in score_pred_dict[node].keys():
            temp_k = {}
            temp_k['pred'] = []
            temp_k['score'] = cos_sims[id]/depth
            for ik, p in enumerate(preds):
                temp_k['pred'] = temp_k['pred'] + score_pred_dict[p][combo[ik]]['pred'] + [combo[ik]]
                temp_k['score'] =  temp_k['score'] + score_pred_dict[p][combo[ik]]['score']
            if temp_k['score']>score_pred_dict[node][k]['score']:
                score_pred_dict[node][k] = temp_k

        else:
            score_pred_dict[node][k]= {}
            score_pred_dict[node][k]['pred'] = []
            score_pred_dict[node][k]['score'] = cos_sims[id]/depth
            for ik, p in enumerate(preds):
                score_pred_dict[node][k]['pred'] = score_pred_dict[node][k]['pred'] + score_pred_dict[p][combo[ik]]['pred'] + [combo[ik]]
                score_pred_dict[node][k]['score'] =  score_pred_dict[node][k]['score'] + score_pred_dict[p][combo[ik]]['score']
    return score_pred_dict

def get_top_k(all_q, return_cos_sim, encoder, corp_emb, tfidf, tfidf_embs, alpha, neighs='all'):
    ini = encoder.encode(all_q, prompt="s2p_query")
    ini = np.array(ini)
    if neighs=='all':
        neighs = list(range(len(corp_emb)))
    cos_sim = cosine_similarity(ini,corp_emb[neighs])
    if tfidf is not None:
        cos_sim_2 = cosine_similarity(tfidf.transform(all_q),tfidf_embs[neighs])
        assert cos_sim.shape == cos_sim_2.shape, "Matrices must have the same shape!"
        cos_sim = (1-alpha)*cos_sim + alpha*cos_sim_2
    else:
        cos_sim = (1-alpha)*cos_sim
    top_k_s = [np.argsort(cos_sim[i])[::-1][:100] for i in range(len(cos_sim))]
    top_k_s = [[neighs[id] for id in k] for k in top_k_s]
    cos_sims = [np.sort(cos_sim[i])[::-1][:100] for i in range(len(cos_sim))]
    if return_cos_sim:
        return top_k_s, cos_sims
    else:
        return top_k_s

def get_top_k_mq(query_embs,all_q,return_cos_sim, encoder, corp_emb, tfidf, tfidf_embs, alpha, n_subgraphs, neighs='all'):
    ini = np.array(query_embs)
    if neighs=='all':
        neighs = list(range(len(corp_emb)))
    cos_sim = cosine_similarity(ini,corp_emb[neighs])
    if tfidf is not None:
        cos_sim_2 = cosine_similarity(tfidf.transform(all_q),tfidf_embs[neighs])
        assert cos_sim.shape == cos_sim_2.shape, "Matrices must have the same shape!"
        cos_sim = (1-alpha)*cos_sim + alpha*cos_sim_2
    else:
        cos_sim = (1-alpha)*cos_sim
    modified_cos_sim = np.max(cos_sim,axis=0).reshape(1,-1)
    top_k_s = [np.argsort(modified_cos_sim[i])[::-1][:2*n_subgraphs] for i in range(len(modified_cos_sim))]
    top_k_s = [[neighs[id] for id in k] for k in top_k_s]
    cos_sims = [np.sort(modified_cos_sim[i])[::-1][:2*n_subgraphs] for i in range(len(modified_cos_sim))]
    if return_cos_sim:
        return top_k_s, cos_sims
    else:
        return top_k_s

def get_node_depth_dag(G):
    topo_order = list(nx.topological_sort(G))
    depth = {n: float('inf') for n in G.nodes}
    
    for n in topo_order:
        if G.in_degree(n) == 0:
            depth[n] = 0
    
    for n in topo_order:
        for neighbor in G.successors(n):
            if depth[neighbor] > depth[n] + 1:
                depth[neighbor] = depth[n] + 1
    
    return depth

def get_chunks_from_KG(G, n_subgraphs, KG, chunk_emb, encoder, corp_text, tfidf, tfidf_embs, hybrid_alpha, org_qtn, query_embs_dict):
    nodes_order = list(nx.topological_sort(G))
    score_pred_dict = {i:{} for i in nodes_order}
    try:
        depth_dict = get_node_depth_dag(G)
        for node in nodes_order:
            depth = depth_dict[node] + 1
            if len(list(G.predecessors(node)))==0: 
                q = G.nodes[node]['question']
                query_embs = [query_embs_dict[q], query_embs_dict[org_qtn]]
                top_k, cos_sims = get_top_k_mq(query_embs,[q,org_qtn],return_cos_sim=1, encoder=encoder, corp_emb=chunk_emb, tfidf=tfidf, tfidf_embs=tfidf_embs, alpha=hybrid_alpha, n_subgraphs=n_subgraphs)
                top_k = top_k[0][:n_subgraphs]
                cos_sims = cos_sims[0][:n_subgraphs]
                score_pred_dict[node] = {k:{'pred':[],'score':cos_sims[id]} for id,k in enumerate(top_k)}
            else:
                preds = list(G.predecessors(node))
                if len(preds)>1:
                    all_combo = list(product(*[list(score_pred_dict[p].keys()) for p in preds]))
                else:
                    all_combo = [[i] for i in score_pred_dict[preds[0]].keys()]

                q = G.nodes[node]['question']
                for idx,combo in enumerate(all_combo):
                    common_neighs = list(set.union(*[set(KG.neighbors(c)) for c in combo]))
                    for i,c in enumerate(all_combo):
                        if i!= idx:
                            common_neighs += [cm for cm in c]
                    common_neighs = list(set(common_neighs))        

                    if len(common_neighs)==0:
                        continue

                    top_k_q, cos_sims_q = get_top_k_mq([query_embs_dict[q]],[q],return_cos_sim=1, encoder=encoder, corp_emb=chunk_emb,
                                            neighs= [combo[i] for i in range(len(combo))], tfidf=tfidf, tfidf_embs=tfidf_embs, alpha = hybrid_alpha, n_subgraphs=n_subgraphs)
                    query_embs = [query_embs_dict[q], query_embs_dict[org_qtn]]
                    top_k, cos_sims = get_top_k_mq(query_embs,[q,org_qtn],return_cos_sim=1, encoder=encoder, corp_emb=chunk_emb,
                                            neighs= common_neighs, tfidf=tfidf, tfidf_embs=tfidf_embs, alpha = hybrid_alpha, n_subgraphs=n_subgraphs)
                   
                    ret_text = ""
                    for i in range(len(combo)):
                        if cos_sims_q[0][i]>np.max(cos_sims[0]):
                            ret_text += corp_text[combo[i]]+"\n" 
                    new_qtn_emb = encoder.encode([ret_text+q], prompt="s2p_query",show_progress_bar = False)[0]
                    query_embs = [new_qtn_emb, query_embs_dict[q], query_embs_dict[org_qtn]]
                    top_k, cos_sims = get_top_k_mq(query_embs,[ret_text+q,q,org_qtn],return_cos_sim=1,
                        encoder=encoder, corp_emb=chunk_emb,neighs= common_neighs, tfidf=tfidf, tfidf_embs=tfidf_embs, alpha = hybrid_alpha, n_subgraphs=n_subgraphs)
                    
                    top_k = top_k[0][:n_subgraphs]
                    cos_sims = cos_sims[0][:n_subgraphs]

                    score_pred_dict = update_score_pred_dict(node, combo, top_k, cos_sims, score_pred_dict, preds, depth)
                
                keep_top_n_subgraphs = sorted(score_pred_dict[node].items(), key=lambda x: -x[1]['score'])[:n_subgraphs]
                score_pred_dict[node] = {k:v for k,v in keep_top_n_subgraphs}
                
                if len(list(G.successors(node)))==0:
                    corpus_id = [] 
                    for k in score_pred_dict[node].keys():
                        corpus_id+=score_pred_dict[node][k]['pred']+[k]
    except:
        q = G.nodes[nodes_order[-1]]['question']
        top_k, cos_sims = get_top_k_mq([query_embs_dict[q]],[q],return_cos_sim=1, encoder=encoder, corp_emb=chunk_emb, tfidf = tfidf, tfidf_embs = tfidf_embs, alpha = hybrid_alpha, n_subgraphs=n_subgraphs)
        top_k = top_k[0][:n_subgraphs]
        corpus_id = top_k

    return corpus_id
        
def browsenet_retriever(questions, split_queries, KG, chunk_emb, encoder, n_subgraphs, corp_text, hybrid_alpha):
    
    if hybrid_alpha>0:
        tfidf = TfidfVectorizer()
        tfidf_embs = tfidf.fit_transform(corp_text)
    else:
        tfidf = None
        tfidf_embs = None
    
    split_queries = [split_queries[q['question']] for q in questions]
    retrieved_corpus = []
   
    for idx,q in tqdm(enumerate(split_queries) ,total=len(split_queries)):
        q_subgraph = get_q_subgraph(q)
        org_qtn = questions[idx]['question']
        all_q = [q_subgraph.nodes[i]['question'] for i in q_subgraph.nodes()] + [org_qtn]
        query_embs = encoder.encode(all_q, prompt="s2p_query",show_progress_bar = False)
        query_embs_dict = {}
        
        for idx, query in enumerate(all_q):
            query_embs_dict[query] = query_embs[idx]
            
        prev_corpus = []
        if len(q_subgraph.edges())==0:
            all_q = [q_subgraph.nodes[i]['question'] for i in q_subgraph.nodes()]
            corpus_temp = []
            for q in all_q:
                query_embs = [query_embs_dict[q], query_embs_dict[org_qtn]]
                corpus_temp.extend(get_top_k_mq(query_embs, [q,org_qtn],0, encoder, chunk_emb, tfidf, tfidf_embs, alpha = hybrid_alpha, n_subgraphs=n_subgraphs))
            corpus = []
            for i in range(n_subgraphs):
                for j in range(len(q_subgraph.nodes())):
                    corpus.append(corpus_temp[j][i]) 
        else:
            for g in nx.connected_components(q_subgraph.to_undirected()): 
                q_sub_subgraph = q_subgraph.subgraph(g)
                if len(q_sub_subgraph.edges())!=0:
                    chunks = get_chunks_from_KG(q_sub_subgraph, n_subgraphs, KG, chunk_emb, encoder, corp_text,tfidf,
                                                 tfidf_embs, hybrid_alpha, org_qtn = org_qtn, query_embs_dict=query_embs_dict)
                    
                    id_ = len(chunks)//n_subgraphs
                    chunks = [chunks[i:i+id_] for i in range(0, len(chunks), id_)]
                    prev_corpus.append(chunks)
                else:
                    all_q = [q_sub_subgraph.nodes[i]['question'] for i in q_sub_subgraph.nodes()]
                    chunks = []
                    for q in all_q:
                        query_embs = [query_embs_dict[q], query_embs_dict[org_qtn]]
                        chunks.extend(get_top_k_mq(query_embs, [q,org_qtn], 0, encoder, chunk_emb, tfidf, tfidf_embs,
                                                    alpha=hybrid_alpha, n_subgraphs=n_subgraphs))
                        
                    chunks = chunks[0] 
                    chunks = [[c] for c in chunks[:n_subgraphs]]
                    prev_corpus.append(chunks)
            corpus=[]
            for i in range(n_subgraphs):
                for j in range(len(prev_corpus)):
                    corpus.append(prev_corpus[j][i]) 
            corpus = flatten(corpus)
        corpus = list(dict.fromkeys(corpus))
        retrieved_corpus.append(corpus)
    return retrieved_corpus

def naiverag_retriever(questions, chunk_emb, encoder, corp_text, hybrid_alpha):
   
    tfidf = TfidfVectorizer()
    tfidf_embs = tfidf.fit_transform(corp_text)
    retrieved_corpus = []
    corpus = get_top_k([q['question'] for q in questions], 0, encoder, chunk_emb, tfidf, tfidf_embs, 
                       alpha = hybrid_alpha)
    for i,c in enumerate(corpus):
        corpus[i] = list(dict.fromkeys(c))
        retrieved_corpus.append(corpus[i])
       
    return retrieved_corpus




