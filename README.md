<h1 align="center"> <img src="images/BN.png" width="30" height="30"> BrowseNet: Graph-Based Associative Memory for Contextual Information Retrieval
</h1>

BrowseNet is a novel Retrieval Augumented Generation (RAG) framework that identifies and leverages the structure in a multi-hop query to traverse on a Graph-of-chunks for information retrieval.

![BrowseNet](images/browseNet.png)

## Setup Environment

Create a conda environment and install dependency:

```shell
conda env create -f environment.yml
```

To match keywords with the similar keywords, ColBERTV2 is required download the pre-trained [checkpoint](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz), extract and put it in the folder 'src/indexer/exp/colbertv2.0'.

```shell
cd src/indexer/exp
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
tar -xvzf colbertv2.0.tar.gz
```
## Format of the dataset
All the benchmark datasets used in this study is available in the folder ./Datasets/. 
To test BrowseNet on a new dataset follow the below steps:
1. Create a folder in ./datasets. The name of the folder has to be the name of the dataset
2. Corpus and the questions have to be uploaded in the specified folder as corpus.json and questions.json respectively.
3. The format of corpus.json has to be as shown below
```json
[
  {
    "title": "<title of the passage>",
    "text": "<passage>"
  },
]
```
4. The format of questions.json has to be as shown below. "gold_ids", "edge_list", and "answer" are optional properties required for evaluation of the pipeline for retrieval, knowledge graph construction and answer generation respectively.
```json
[
  {
    "question": "<multi-hop query>",
    "gold_ids": "<list of indices of corpus required to answer the question>",
    "edge_list": "<list of edges in query-subgraph>",
    "answer": "<answer to the question>"
  },
]
```

Parameters required to be defined are:
- OPENAI_API_KEY: Open AI API key required to be defined to use the OPENAI models
- DEEPSEEK_API_KEY: Deepseek API key requierd to be defined to use the deepseek models (Optional)
- DEVICE: 'cuda' or 'cpu'. This defines the device to load the encoder model
- DATASET: '2wikimqa' or 'hotpoqa' or 'musique'. This defines the name of the dataset
- ALPHA: 0-1. This defines the weightage to be provided for sparse encoder like 'TF_IDF'. In this study we have used 0 for all the results
- RETRIEVAL_METHOD: 'browsenet' or 'naiverag'
- NER_MODEL: 'gliner' or 'gpt-4o'. This defines the type of model to use for NER
- SEM_MODEL: 'miniLM' or 'stella' or 'nvembedv2' or 'granite' or 'qwen2'. Model to use to generate the dense embeddings
- N_SUBGRAPHS: 5. The parameter that defined number of subgraphs to be retrieved. Larger numbers require increased context size.
- SUBQUERY_MODEL: 'gpt-4o' or 'o4-mini' or 'deepseek-reasoner'
- COLBERT_THRESHOLD: 0.9. This defines the synonymity threshold to get the similar words.
- LLM: 'openai' or 'deepseek'. LLM to use for QA
- MODEL: 'gpt-4o' or 'gpt-3.5-turbo'. The model to choose for QA
- N_CHUNKS: 5. Number of chunks to be provided as context for answer generation.


All the above parameters have to be stored as environment variables. The sample .env file is provided in the repo


Once the above parameters are defined, then running, main.py file should index, retrieve and generate answers for the provided dataset.
