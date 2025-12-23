from sentence_transformers import SentenceTransformer

class NVEmbedEncoder:
    def __init__(self,device):
        self.model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True).to(device)
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"
        self.task_instruction = {
            "query": "Given a question, retrieve passages that answer the question",
        }

    def _add_eos(self, texts):
        eos_token = self.model.tokenizer.eos_token
        return [text + eos_token for text in texts]

    def encode(self, texts, prompt="passage", batch_size=2, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        if "query" in prompt:
            prefix = f"Instruct: {self.task_instruction['query']}\nQuery: "
            texts = [prefix + text for text in texts]
        
        texts = self._add_eos(texts)
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, **kwargs)
    
class QwenEncoder:
    def __init__(self,device):
        self.model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
        # In case if you want to reduce the maximum length:
        # self.model.max_seq_length = 8192
        self.model.tokenizer.padding_side = "right"
        self.model.to(device)
    
    def encode(self,texts,prompt="passage",**kwargs):
        if isinstance(texts, str):
            texts = [texts]
        if "query" in prompt:
            return self.model.encode(texts, prompt="query",**kwargs)
        
        return self.model.encode(texts,**kwargs)