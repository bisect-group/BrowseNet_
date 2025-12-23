from dotenv import load_dotenv
load_dotenv()

import os
import logging
import time
import json
import torch
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.retriever import NVEmbedEncoder, QwenEncoder
from src.BrowseNet import BrowseNet
from src.NaiveRAG import NaiveRAG

ROOT_DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    
    dataset = os.environ['DATASET']
    alpha = float(os.environ['ALPHA'])
    ner_model = os.environ['NER_MODEL']
    sem_model = os.environ['SEM_MODEL']
    n_subgraphs = int(os.environ['N_SUBGRAPHS'])
    subquery_model = os.environ['SUBQUERY_MODEL']
    colbert_threshold = float(os.environ['COLBERT_THRESHOLD'])
    retrieval_method = os.environ['RETRIEVAL_METHOD']
    n_chunks = int(os.environ['N_CHUNKS'])
    llm = os.environ['LLM']
    model_name = os.environ['MODEL']

    os.makedirs(f'logs/{dataset}', exist_ok=True)
    os.makedirs(f'results/{dataset}', exist_ok=True)
    os.makedirs(f'artifacts/{dataset}', exist_ok=True)

    logging.basicConfig(level=logging.INFO, filename='logs/{}/{}.log'.format(dataset,datetime.now().timestamp()),
                            filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)  

    device = os.getenv('DEVICE')
    device = device if 'cuda' in device and torch.cuda.is_available() else 'cpu'

    logger.info(f"::> PIPELINE INFO:\n\
                Dataset: {dataset}\n\
                Retrieval Method: {retrieval_method}\n\
                Sem Model: {sem_model}\n\
                Device: {device}\n\
                NER Model: {ner_model}\n\
                Subquery Model: {subquery_model}\n\
                Colbert Threshold: {colbert_threshold}\n\
                Number of Subgraphs: {n_subgraphs}\n\
                Alpha: {alpha}\n\
                Number of Chunks: {n_chunks}\n\
                ")

    start_time = time.time()
    
    logger.info(f"::> Time taken to load the encoder: {time.time()-start_time} seconds")

    if retrieval_method.lower() == 'browsenet':
        logger.info(f"::> Initializing BrowseNet...")
        browsenet = BrowseNet(
            dataset = dataset,
            device=device,
            ner_model = ner_model,
            sem_model= sem_model,
            subquery_model = subquery_model,
            colbert_threshold = colbert_threshold,
            n_subgraphs = n_subgraphs,
            alpha = alpha
        )

        logger.info(f"::> Indexing...")
        browsenet.index()

        questions = json.load(open(ROOT_DIR / 'datasets' / dataset / 'questions.json','r'))

        logger.info(f"::> Total Questions: {len(questions)}")
        logger.info(f"::> Starting Retrieval...")
        split_queries, retrieved_corpus = browsenet.retrieve(questions)

        logger.info(f"::> Starting Retrieval Evaluation...")
        result_dict = browsenet.retrieval_eval(questions, retrieved_corpus)

        logger.info(f"::> Starting QA...")
        browsenet.qa(questions, split_queries, retrieved_corpus, n_chunks, llm, model_name)

        logger.info(f"::> Starting QA Evaluation...")
        em, f1 = browsenet.qa_eval(dataset, n_chunks)

    else:
        logger.info(f"::> Initializing NaiveRAG...")
        naiverag = NaiveRAG(
            dataset = dataset,
            device=device,
            sem_model= sem_model,
            alpha = alpha
        )

        logger.info(f"::> Indexing...")
        naiverag.index()

        questions = json.load(open(ROOT_DIR / 'datasets' / dataset / 'questions.json','r'))

        logger.info(f"::> Total Questions: {len(questions)}")
        logger.info(f"::> Starting Retrieval...")
        split_queries,retrieved_corpus = naiverag.retrieve(questions)

        logger.info(f"::> Starting Retrieval Evaluation...")
        result_dict = naiverag.retrieval_eval(questions, retrieved_corpus)

        logger.info(f"::> Starting QA...")
        naiverag.qa(questions, split_queries, retrieved_corpus, n_chunks, llm, model_name)

        logger.info(f"::> Starting QA Evaluation...")
        em, f1 = naiverag.qa_eval(dataset, n_chunks)

    logger.info(f"::> Total time taken for the pipeline: {time.time() - start_time} seconds.\n\n\n")