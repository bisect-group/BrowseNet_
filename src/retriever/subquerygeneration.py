from openai import OpenAI
import json
from ..prompts.sq_gen_prompts import init_prompt, musique_few_shot_demo, wikimqa_few_shot_demo, hotpot_few_shot_demo
import pickle as pkl
from pathlib import Path
import os

FILE_DIR = Path(__file__).resolve().parent
FILE_DIR_PARENT = FILE_DIR.parent
ROOT_DIR = FILE_DIR_PARENT.parent

def query_openai_model(prompt, query, client, model="gpt-4o"):
    try:
        if model=='o4-mini':
            response = client.chat.completions.create(
            model=model, 
            messages=[{"role": "system", "content": f"{prompt}"},
                      {"role": "user", "content": f"{query}"}],
                      stream=False
        )
        else:
            response = client.chat.completions.create(
                model=model, 
                messages=[{"role": "system", "content": f"{prompt}"},
                        {"role": "user", "content": f"{query}"}],
                max_tokens=1500,
                temperature=0
            )
        return response
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return None

def process_queries(prompt, queries, model, dataName, client):
    results_path = ROOT_DIR / 'artifacts' / f'{dataName}' / f'keyword_subquery_representation_{model}.json'
    if os.path.exists(results_path):
        print(f"Results already exist at {results_path}.")
        split_queries = json.load(open(results_path, 'r'))
        return split_queries
    
    results = {}
    responses = {}
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: {query}")
        response = query_openai_model(prompt, query, client, model)
        ans = response.choices[0].message.content
        results[query] = ans
        responses[query] = response
        print(f"Response: {ans}\n")

        response_pickle_path = ROOT_DIR / 'artifacts' / f'{dataName}' /f'keyword_subquery_representation_{model}.pkl'
        with open(response_pickle_path, 'wb') as f:
            pkl.dump(responses, f)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

    print(f"All queries processed. Results saved to {results_path}")
    return results

def get_subqueries(dataset, subquery_model, questions):

    dataName = dataset
    model = subquery_model

    if 'hotpotqa' in dataName:
        few_shot_demo = hotpot_few_shot_demo
    elif 'musique' in dataName:
        few_shot_demo = musique_few_shot_demo
    elif '2wikimqa' in dataName:
        few_shot_demo = wikimqa_few_shot_demo

    prompt = f"{init_prompt}\n{few_shot_demo}"
    if model=='deepseek-reasoner':
        API_KEY = os.environ.get("DEESEEK_API_KEY")
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    else:
        API_KEY = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=API_KEY)
    queries = [q['question'] for q in questions]
    split_queries = process_queries(prompt, queries, model, dataName, client)
    return split_queries