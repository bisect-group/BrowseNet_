'''The functions: get_ner_prompt, named_entity_recognition_openai, init_langchain_model are adopted from the HippoRAG project.'''
import os
from collections import defaultdict
from gliner import GLiNER
import pickle as pkl
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from pathlib import Path
from ..helpers.utils import preprocess_entity
from ..prompts.ner_prompt import get_ner_prompt

FILE_DIR = Path(__file__).resolve().parent
FILE_DIR_PARENT = FILE_DIR.parent
ROOT_DIR = FILE_DIR_PARENT.parent 

def named_entity_recognition_openai(client, passage: str):
    ner_prompts = get_ner_prompt()
    ner_messages = ner_prompts.format_prompt(user_input=passage)

    not_done = True

    total_tokens = 0
    response_content = '{}'

    while not_done:
        try:
            if isinstance(client, ChatOpenAI):  # JSON mode
                chat_completion = client.invoke(ner_messages.to_messages(), temperature=0, response_format={"type": "json_object"})
                response_content = chat_completion.content
                response_content = eval(response_content)
                total_tokens += chat_completion.response_metadata['token_usage']['total_tokens']

            if 'named_entities' not in response_content:
                response_content = []
            else:
                response_content = response_content['named_entities']

            not_done = False
        except Exception as e:
            print('Passage NER exception')
            print(e)

    return response_content, total_tokens

def init_langchain_model(model_name: str, temperature: float = 0.0, max_retries=5, timeout=60, **kwargs):
    from langchain_openai import ChatOpenAI
    assert model_name.startswith('gpt-')
    return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=model_name, temperature=temperature, max_retries=max_retries, timeout=timeout, **kwargs)

def extract_keywords(dataset, chunks_dict, ner_method='gliner', device='cuda'):
    entities_dict_path = ROOT_DIR / f'artifacts/{dataset}/entities_dict_{ner_method}.pkl'
    if os.path.exists(entities_dict_path):
        print("Entities dict already exists!")
        return
    if ner_method == 'gliner':
        gliner_model = GLiNER.from_pretrained("urchade/gliner_large-v2.1").to(device)
        labels=['event','facilities','language','location','money','nationality','religious','political','organization',
                'person','product','work_of_art', 'occupation','time','ordinal','date']
    else:
        client = init_langchain_model(ner_method) 
    kw2chunk = defaultdict(set) #kw1 -> [chunk1, chunk2, ...]
    chunk2kw = defaultdict(set) #chunk -> [kw1, kw2, ...]
    id2chunk = defaultdict(int)
    for i,chunk_dict in tqdm(enumerate(chunks_dict),total=len(chunks_dict)):
        chunk = chunk_dict['title']+'\n'+chunk_dict['text']

        if ner_method=='gliner' and len(chunk.split(" "))>200:
            chunks = [chunk[i:i+200] for i in range(0, len(chunk), 200)]
        else:
            chunks = [chunk]

        entities = []
        for chunk0 in chunks:
            if ner_method=='gliner':
                entities += gliner_model.predict_entities(chunk0,labels)
            else:
                response_content, total_tokens = named_entity_recognition_openai(client, chunk0)
                entities += response_content
        if ner_method=='gliner':
            entities = [i['text'] for i in entities]

        id2chunk[i] = chunk
        for entity in entities:
            clean_entity = preprocess_entity(entity)
            if clean_entity == '': 
                continue
            kw2chunk[clean_entity].add(i)
            chunk2kw[i].add(clean_entity)
        
    for key in kw2chunk:
        kw2chunk[key] = list(kw2chunk[key])

    for key in chunk2kw:
        chunk2kw[key] = list(chunk2kw[key])
    kws_set = set()
    for _,kws in chunk2kw.items():
        for kw in kws:
            kws_set.add(kw)
    all_kws = list(kws_set)
    entities_dict = {'kw2chunk':kw2chunk,'chunk2kw':chunk2kw, 'id2chunk':id2chunk, 'all_kws':all_kws}
    print("Extracted entities dict and writing to a file...")
    pkl.dump(entities_dict, open(entities_dict_path, 'wb'))
    return entities