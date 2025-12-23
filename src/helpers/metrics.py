from .utils import clean_text
from collections import Counter

def get_recall(queries, top_ks):
    """
    queries: list of query dicts with 'gold_ids' key
    top_ks: list of list of retrieved ids for each query

    returns: list of recall@2, recall@5, recall@10, recall@20, recall@overall
    """
    recall2 = [len([i for i in top_ks[i][:2] if i in q['gold_ids']])/len(q['gold_ids']) for i,q in enumerate(queries)]
    recall5 = [len([i for i in top_ks[i][:5] if i in q['gold_ids']])/len(q['gold_ids']) for i,q in enumerate(queries)]
    recall10 = [len([i for i in top_ks[i][:10] if i in q['gold_ids']])/len(q['gold_ids']) for i,q in enumerate(queries)]
    recall20 = [len([i for i in top_ks[i][:20] if i in q['gold_ids']])/len(q['gold_ids']) for i,q in enumerate(queries)]
    recall_overall = [len([i for i in top_ks[i] if i in q['gold_ids']])/len(q['gold_ids']) for i,q in enumerate(queries)]
    return [recall2, recall5, recall10, recall20, recall_overall]

def exact_match_score(generated_answer, correct_answer):
    """
    generated_answer: str 
    correct_answer: str
    returns: 1.0 if exact match else 0.0
    """
    return 1.0 if clean_text(generated_answer) == clean_text(correct_answer) else 0.0

def f1_score(generated_answer, correct_answer):
    """
    generated_answer: str
    correct_answer: str
    returns: f1 score between generated_answer and correct_answer
    """
    generated_answer = clean_text(generated_answer)
    correct_answer = clean_text(correct_answer)
    
    common_words = Counter(generated_answer.split()) & Counter(correct_answer.split())
    no_of_common_words = sum(common_words.values())
    if no_of_common_words == 0:
        return 0
    precision = no_of_common_words / len(generated_answer.split())
    recall = no_of_common_words / len(correct_answer.split())
    f1 = (2 * precision * recall) / (precision + recall)
    return f1