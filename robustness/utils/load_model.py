import torch
import transformers
from transformers import AutoModel,AutoTokenizer
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from torch import Tensor
import os
# SentenceTransformer generated embeddings are default mean pooling, only reasonir and contriever are mean pooling

def load_model_hf(model_name):
    # Except for contriever, all embedding models are set to 8192 maximum output length.

    model_kwargs = {
    "torch_dtype": torch.bfloat16,  # Using bfloat16, more stable and better performance
    # "attn_implementation": "flash_attention_2",  # Using Flash Attention 2 to improve efficiency
    }

    if model_name=='reasonir':
        # encoder = SentenceTransformer("reasonir/ReasonIR-8B", trust_remote_code=True, model_kwargs=model_kwargs)
        # encoder.max_seq_length = 8192
        tokenizer = AutoTokenizer.from_pretrained("reasonir/ReasonIR-8B")
        model = AutoModel.from_pretrained("reasonir/ReasonIR-8B", torch_dtype=torch.bfloat16, trust_remote_code=True,cache_dir=os.getenv('HF_HOME'))
        encoder = HFtoSF(model, tokenizer,  normalize=True , pooling='mask_prompt_mean', device='cuda')


    elif model_name == 'contriever':
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco") # this is the latest version
        model = AutoModel.from_pretrained("facebook/contriever-msmarco",torch_dtype=torch.bfloat16 , trust_remote_code=True) #torch_dtype=torch.bfloat16
        encoder = HFtoSF(model, tokenizer, normalize=False , pooling='mean', max_seq_length = 512, device='cuda') # use dot product

    elif model_name=='bge_reasoner':
        # last token
        # tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reasoner-embed-qwen3-8b-0923")
        # model = AutoModel.from_pretrained("BAAI/bge-reasoner-embed-qwen3-8b-0923", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928") # this is the latest version
        model = AutoModel.from_pretrained("hanhainebula/reason-embed-qwen3-8b-0928", torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer, normalize=True , pooling='last', device='cuda')

    elif model_name=='diver':
        tokenizer = AutoTokenizer.from_pretrained('AQ-MedAI/Diver-Retriever-4B', padding_side='left')
        model = AutoModel.from_pretrained('AQ-MedAI/Diver-Retriever-4B',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,normalize=True , pooling='last', device='cuda')


    elif model_name=='qwen3':
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left')
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,   normalize=True ,pooling='last', device='cuda')
    elif model_name=='qwen3_0.6B':
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,   normalize=True ,pooling='last', device='cuda')
    elif model_name=='qwen3_4B':
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left')
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,   normalize=True ,pooling='last', device='cuda')

    elif model_name=='linq':
        tokenizer = AutoTokenizer.from_pretrained('Linq-AI-Research/Linq-Embed-Mistral')
        model = AutoModel.from_pretrained('Linq-AI-Research/Linq-Embed-Mistral',torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer,  normalize=True , pooling='last', device='cuda')

    elif model_name == 'gte':
        tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True,torch_dtype=torch.bfloat16)
        encoder = HFtoSF(model, tokenizer, normalize=True , pooling='last', device='cuda')

    elif model_name=='bge_m3':
        model = AutoModel.from_pretrained('BAAI/bge-m3', trust_remote_code=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        encoder = HFtoSF(model, tokenizer, normalize=True ,pooling='cls',  device='cuda')

    else:
        raise Exception('model_name error')

    return encoder,tokenizer

# There is also a code to manually convert the HF model to SF format, which is actually defined as a class to implement encode

class HFtoSF:
    def __init__(self, hf_model, hf_tokenizer,normalize= False, pooling='last', max_seq_length=8192 ,device='cuda'):
        try:
            self.hf_model = hf_model.to(device)
            self.hf_model.eval()
        except: # for BGE M3, no attribute 'eval' and 'to'
            self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        self.device = device
        self.max_seq_length = max_seq_length  # Adjust according to the specific model
        self.pooling = pooling
        self.normalize = normalize

    def _pooling(self, last_hidden_state, attention_mask, prompt=None):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['mask_prompt_mean']: # Follow reasonir's pooling method, remove the influence of prompt tokens and then average
            if prompt is None:
                prompt = self.prompt
            self.prompt_tokens = self.hf_tokenizer( prompt, padding=False, add_special_tokens=True, return_tensors='pt')
            attention_mask[:, :len(self.prompt_tokens['input_ids'][0])] = 0
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
 
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def tokenize_texts(self, texts):
        batch_dict = self.hf_tokenizer(texts, max_length=self.max_seq_length, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, pad_to_multiple_of=8)
        batch_dict = { k: v.to(self.device) for k, v in batch_dict.items() }
        return batch_dict

    def encode(self, texts, prompt, convert_to_numpy=False, show_progress_bar=False):
        # The input is a list of texts that have already been through batch size
        # The output is the corresponding embeddings

        self.prompt = prompt  # Each time encode is called, prompt is fixed, so it can be set here
        if isinstance(texts, str):
            texts = [texts]
        # Concatenate prompt to texts
        if prompt is not None and prompt != '':
            texts = [prompt + text for text in texts]
        batch_inputs = self.tokenize_texts(texts)
        with torch.no_grad():
            outputs = self.hf_model(**batch_inputs)
            embeddings = self._pooling(outputs.last_hidden_state, batch_inputs['attention_mask'])
        if convert_to_numpy:
            embeddings = embeddings.cpu().numpy()
        return embeddings