import time
import logging
from pathlib import Path
import os 
import json
import pickle as pkl
import numpy as np
from sentence_transformers import SentenceTransformer
from .retriever import naiverag_retriever
from .retriever import QwenEncoder, NVEmbedEncoder
from .qa import save_answers
from .helpers.metrics import get_recall

FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent

logger = logging.getLogger(__name__)

class NaiveRAG:
    def __init__(self, dataset, device, sem_model, alpha):
        self.dataset = dataset
        self.device = device
        self.sem_model = sem_model
        self.alpha = alpha
        self.encoder = None
        self.chunks_dict = json.load(open(ROOT_DIR / 'datasets' / dataset / 'corpus.json', 'r'))
        self.corp_text = [c['title']+'\n'+c['text'] for c in self.chunks_dict]

        self.chunk_embs = None

    def index(self):
        global_start = time.time()
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

        st = time.time()
        embs_path = ROOT_DIR / f'artifacts/{self.dataset}/sem_embs_{self.sem_model}.pkl'
        if os.path.exists(embs_path):
            self.chunk_embs = pkl.load(open(embs_path,'rb'))
        else:
            print("Getting Embs!")
            self.chunk_embs = self.encoder.encode(self.corp_text, show_progress_bar=True)
            os.makedirs(ROOT_DIR / 'artifacts' / self.dataset, exist_ok=True)
            pkl.dump(self.chunk_embs, open(embs_path,'wb'))
        logger.info(f"::> Chunk embeddings ready. Time taken: {time.time() - st} seconds.")

        logger.info(f"::> Total time taken for indexing: {time.time() - global_start} seconds.\n\n\n")
        return

    def retrieve(self,questions):
        global_start = time.time()
        start_time = time.time()

        results_path = ROOT_DIR / f'results/{self.dataset}/results{self.alpha}_NaiveRAG_{self.sem_model}.json'
        if os.path.exists(results_path):
            print('Results already exist')
            with open(results_path,'r') as f:
                retrieved_corpus = json.load(f)
        else:
            retrieved_corpus = naiverag_retriever(questions, self.chunk_embs, self.encoder,
                                            corp_text=self.corp_text, hybrid_alpha = self.alpha)
            os.makedirs(f"results/{self.dataset}", exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(retrieved_corpus, f, indent=4)
        logger.info(f"::> Retrieval Completed Successfully in {time.time()-start_time} seconds")
        logger.info(f"::> Total time taken for retrieval: {time.time() - global_start} seconds.\n\n\n")
        return None,retrieved_corpus
    
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