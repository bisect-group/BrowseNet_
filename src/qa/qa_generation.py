import os
import json
import pickle as pkl
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from ..prompts.rag_qa_prompts import BrowseNetQAPrompts, NaiveRAGQAPrompts
from ..helpers.metrics import exact_match_score, f1_score

retrieval_method = os.getenv('RETRIEVAL_METHOD') 

FILE_DIR = Path(__file__).resolve().parent
FILE_DIR_PARENT = FILE_DIR.parent
ROOT_DIR = FILE_DIR_PARENT.parent

def generate_answer(idx,rd,corpus,questions,n_chunks,client,dataset,split_queries):

    file_path = ROOT_DIR / 'results' / 'generation_cache' / f'{dataset}_{n_chunks}' / f"{idx}.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            response_content = json.load(f)["generated_response"]
        return idx, response_content

    if(retrieval_method.lower()=='browsenet'):
        one_shot_rag_qa_input = BrowseNetQAPrompts.one_shot_rag_qa_input_prompt()
        one_shot_rag_qa_output = BrowseNetQAPrompts.one_shot_rag_qa_output_prompt()
        answer_generation_prompt = BrowseNetQAPrompts.answer_generation_prompt()
    else:
        one_shot_rag_qa_input = NaiveRAGQAPrompts.one_shot_rag_qa_input_prompt()
        one_shot_rag_qa_output = NaiveRAGQAPrompts.one_shot_rag_qa_output_prompt()
        answer_generation_prompt = NaiveRAGQAPrompts.answer_generation_prompt()

    user_prompt = ''
    q = rd[idx][:n_chunks]
    
    user_prompt += f'\n\nQuestion : {questions[idx]["question"]}\n'

    if retrieval_method.lower()=='browsenet':
        user_prompt += f"\n\nSubqueries: {split_queries[idx]}\n"

    user_prompt += f"\n\nRetrieved Context:\n\n"
    for chunkid in q:
        user_prompt += f"""Wikipedia Title: {corpus[chunkid]['title']}\n{corpus[chunkid]['text']}\n"""
    
    messages = [
        SystemMessage(content=answer_generation_prompt),
        HumanMessage(content=one_shot_rag_qa_input),
        AIMessage(content=one_shot_rag_qa_output),
        HumanMessage(user_prompt)
    ]
    
    try:
        chat_completion = client.invoke(messages)
        response_content = chat_completion.content
    except Exception as e:
        response_content = f"Error Occurred! {e}"

    # Save the response
    with open(file_path, "w") as f:
        json.dump({"generated_response": response_content}, f)

    return idx,response_content


def save_answers(dataset,corpus,questions,rd,n_chunks,llm,model_name,split_queries,num_threads=32):
    '''
    dataset: str, name of the dataset
    corpus: list, list of documents
    questions: list, list of questions
    rd: list, list of retrieved chunks for each question
    n_chunks: int, number of chunks to pass to the llm
    llm: str, name of the llm to use (eg: 'openai')
    model_name: str, name of the llm model to use (eg: 'gpt-3.5-turbo')
    split_queries: list, list of subqueries for each question
    '''
    
    file_path = ROOT_DIR / 'results' / 'generation_cache' / f'{dataset}_{n_chunks}' / 'result.pkl' 
    if(os.path.exists(file_path)):
        print("QA already exist!")
        return
    
    os.makedirs(file_path.parent,exist_ok=True)

    if llm=='openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Please set OPENAI_API_KEY in the environment variables.")
        model = ChatOpenAI(api_key = api_key, model = model_name, temperature = 0.0, max_retries = 10, timeout = 100)
    elif llm=='deepseek':
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            print("Please set DEEPSEEK_API_KEY in the environment variables.")
        model = ChatOpenAI(api_key = api_key, base_url="https://api.deepseek.com", model = model_name, temperature = 0.0, max_retries = 10, timeout = 100)

    result = {}
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(generate_answer, idx,rd,corpus,questions,n_chunks,model,dataset,split_queries) for idx in range(len(questions))]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Answers"):
            idx,response_content = future.result()
            try:
                result[idx] = {}
                result[idx]['generated_response'] = response_content
                generated_answer = response_content.split("Answer: ")[-1].strip()
                result[idx]['generated_answer'] = generated_answer
                generated_answers = [a.strip() for a in generated_answer.split("(or)")]
                # print(generated_answer)
                possible_answers = [questions[idx]['answer']]
                if 'answer_aliases' in questions[idx]:
                    possible_answers += questions[idx]['answer_aliases']
                # print(possible_answers)
                max_em_score = 0
                max_f1_score = 0
                for ans in possible_answers:
                    for gans in generated_answers:
                        max_em_score = max(max_em_score,exact_match_score(ans,gans))
                        max_f1_score = max(max_f1_score,f1_score(ans,gans))
                result[idx]['em_score'] = max_em_score
                result[idx]['f1_score'] = max_f1_score
            except Exception as e:
                print(f"Error in processing index {idx}: {e}")
                print(idx,response_content)
        
    pkl.dump(result,open(file_path,'wb'))
    return 