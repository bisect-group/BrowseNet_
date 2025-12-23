import time
import logging
from pathlib import Path
import os 
import json
import pickle as pkl
import numpy as np
from sentence_transformers import SentenceTransformer
from .indexer.NER import extract_keywords
from .indexer.kg_construct import get_colbert_similar_keywords, construct_graph
from .indexer.colbertv2_knn import make_entity_csv, get_nearest_neighbors
from .retriever import browsenet_retriever, get_subqueries
from .retriever import QwenEncoder, NVEmbedEncoder 
from .qa import save_answers
from .helpers.metrics import get_recall

FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent

logger = logging.getLogger(__name__)

class BrowseNet:
    def __init__(self, dataset, device, ner_model, sem_model, subquery_model, colbert_threshold, n_subgraphs, alpha):
        self.dataset = dataset
        self.device = device
        self.ner_model = ner_model
        self.sem_model = sem_model
        self.encoder = None
        self.subquery_model = subquery_model
        self.colbert_threshold = colbert_threshold
        self.n_subgraphs = n_subgraphs
        self.alpha = alpha

        self.chunks_dict = json.load(open(ROOT_DIR / 'datasets' / dataset / 'corpus.json', 'r'))
        self.corp_text = [c['title']+'\n'+c['text'] for c in self.chunks_dict]
        self.chunk_embs = None
        self.KG = None

    
    def _ner(self):
        extract_keywords(self.dataset, self.chunks_dict, self.ner_model, self.device)
        return

    def _entity_linking(self):
        make_entity_csv(self.dataset, self.ner_model)
        get_nearest_neighbors(self.dataset, self.ner_model)
        return

    def _kg_construction(self):
        construct_graph(self.dataset, self.chunks_dict, self.ner_model, self.colbert_threshold)
        return 

    def index(self):
        global_start = time.time()

        st = time.time()
        print("Extracting keywords...")    
        self._ner()
        logger.info(f"::> Keywords extracted. Time taken: {time.time() - st} seconds.")

        st = time.time()
        print("Entity linking...")
        self._entity_linking()
        logger.info(f"::> Entity linking done. Time taken: {time.time() - st} seconds.")

        st = time.time()
        print("Extracting similar keywords with threshold: ", self.colbert_threshold)
        get_colbert_similar_keywords(self.dataset, self.ner_model, self.colbert_threshold)
        logger.info(f"::> Similar keywords extracted. Time taken: {time.time() - st} seconds.")

        st = time.time()
        print("Constructing graph...")
        self._kg_construction()
        logger.info(f"::> Graph constructed. Time taken: {time.time() - st} seconds.")
        logger.info(f"::> Total time taken for indexing: {time.time() - global_start} seconds.\n\n\n")
        
        self.KG = pkl.load(open(ROOT_DIR / f'artifacts/{self.dataset}/graph_keyword_{self.ner_model}_colbert_{self.colbert_threshold}.pkl','rb'))
        
        st = time.time()
        logger.info(f"::> Loading Encoder...")
        if self.sem_model == 'miniLM':
            self.encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1').to(self.device)
        elif self.sem_model == 'stella':
            self.encoder = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).to(self.device)
        elif self.sem_model == 'granite':
            self.encoder = SentenceTransformer("ibm-granite/granite-embedding-125m-english").to(self.device)
        elif self.sem_model == 'nvembedv2':
            self.encoder = NVEmbedEncoder(device=self.device)
        elif self.sem_model == 'qwen2':
            self.encoder = QwenEncoder(device=self.device)
        logger.info(f"::> Encoder Loaded Successfully in {time.time()-st} seconds")

        embs_path = ROOT_DIR / f'artifacts/{self.dataset}/sem_embs_{self.sem_model}.pkl'
        if os.path.exists(embs_path):
            self.chunk_embs = pkl.load(open(embs_path,'rb'))
        else:
            print("Getting Embs!")
            self.chunk_embs = self.encoder.encode(self.corp_text, show_progress_bar=True)
            os.makedirs(ROOT_DIR / 'artifacts' / self.dataset, exist_ok=True)
            pkl.dump(self.chunk_embs, open(embs_path,'wb'))
        logger.info(f"::> Chunk embeddings ready. Time taken: {time.time() - st} seconds.")

        return

    def _query_subgraph_generation(self,questions):
        start_time = time.time()
        print("Generating subqueries...")
        split_queries = get_subqueries(self.dataset, self.subquery_model, questions)
        logger.info(f"::> Subqueries generated. Time taken: {time.time() - start_time} seconds.")
        return split_queries

    def retrieve(self,questions):
        global_start = time.time()
        start_time = time.time()
        split_queries = self._query_subgraph_generation(questions)
        logger.info(f"::> Subquery generation completed in {time.time()-start_time} seconds")

        start_time = time.time()
        results_path = ROOT_DIR / f'results/{self.dataset}/results{self.alpha}_BrowseNet_{self.ner_model}_{self.sem_model}_{self.n_subgraphs}_{self.colbert_threshold}_{self.subquery_model}.json'
        if os.path.exists(results_path):
            print('Results already exist')
            with open(results_path,'r') as f:
                retrieved_corpus = json.load(f)
        else:
            retrieved_corpus = browsenet_retriever(questions, split_queries, self.KG, self.chunk_embs, self.encoder, n_subgraphs=self.n_subgraphs,
                                                    corp_text=self.corp_text, hybrid_alpha = self.alpha)
            with open(results_path, 'w') as f:
                json.dump(retrieved_corpus, f, indent=4)
        logger.info(f"::> Retrieval Completed Successfully in {time.time()-start_time} seconds")
        logger.info(f"::> Total time taken for retrieval: {time.time() - global_start} seconds.\n\n\n")
        return split_queries, retrieved_corpus
    
    def retrieval_eval(self, questions, retrieved_corpus):
        result = get_recall(questions, retrieved_corpus)
        logger.info(f"::> ===================================================== ")
        logger.info(f"::> BrowseNet Retrieval Results: ")
        logger.info(f"::> Recall@2: {sum(result[0])/len(result[0])}")
        logger.info(f"::> Recall@5: {sum(result[1])/len(result[1])}")
        logger.info(f"::> Recall@10: {sum(result[2])/len(result[2])}")
        logger.info(f"::> Recall@20: {sum(result[3])/len(result[3])}")
        logger.info(f"::> Recall Overall: {sum(result[4])/len(result[4])}")
        ks = ["2","5","10","20","overall"]
        result_dict = {f"recall@{ks[i]}": sum(result[i])/len(result[i]) for i in range(len(ks))}
        return result_dict

    def qa(self, questions, split_queries, retrieved_corpus, n_chunks, llm, model_name):
        start_time = time.time()
        split_queries = [split_queries[q['question']] for q in questions]
        save_answers(self.dataset,self.chunks_dict,questions,retrieved_corpus,n_chunks,llm,model_name,split_queries)
        logger.info(f"::> QA Completed Successfully in {time.time()-start_time} seconds")
        return
    
    def qa_eval(self,dataset,n_chunks):
        file_path = ROOT_DIR / 'results' / 'generation_cache' / f'{dataset}_{n_chunks}' / 'result.pkl'
        results = pkl.load(open(file_path, 'rb'))
        em = np.mean([results[i]['em_score'] for i in range(len(results))])
        f1 = np.mean([results[i]['f1_score'] for i in range(len(results))])
        logger.info(f"::> ===================================================== ")
        logger.info(f"::> QA Evaluation Results: ")
        logger.info(f"::> EM: {em}")
        logger.info(f"::> F1: {f1}")
        return em, f1