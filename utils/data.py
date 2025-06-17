import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class NCEDataset(Dataset):
    def __init__(self, id, tokens, styolometric_features):
        super().__init__()
        self.id = id
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        
        self.styolometric_features = torch.from_numpy(styolometric_features).float()
        self.id_set = set(self.id)
        self.id_list = list(self.id_set)
        self.id_to_indices = {id: np.where(self.id == id)[0]
                                    for id in self.id_set}

        
    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, author_id):
        author_indices = self.id_to_indices[self.id_list[author_id]]
        idx = np.random.choice(author_indices, 2, replace=False)

        anchor_input_ids = self.input_ids[idx[0]]
        anchor_attention_mask = self.attention_mask[idx[0]]
        anchor_id = self.id[idx[0]]
        anchor_styolometric_features = self.styolometric_features[idx[0]]

        # Get a positive sample
        positive_input_ids = self.input_ids[idx[1]]
        positive_attention_mask = self.attention_mask[idx[1]]
        positive_styolometric_features = self.styolometric_features[idx[1]]

        return {
            'anchor_id': anchor_id,
            'anchor_input_ids': anchor_input_ids,
            'anchor_attention_mask': anchor_attention_mask,
            'anchor_styolometric_features': anchor_styolometric_features,
            'positive_input_ids': positive_input_ids,
            'positive_attention_mask': positive_attention_mask,
            'positive_styolometric_features': positive_styolometric_features
        }
    
def tokenize_df(tokenizer, texts, max_length=512):
    # Split into chunks of 512 texts
    for i in tqdm(range(0, len(texts), 512)):
        chunk = texts[i:i+512]
        chunk = tokenizer(chunk.to_list(), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        if i == 0:
            train_tokens = chunk
        else:
            train_tokens['input_ids'] = torch.cat((train_tokens['input_ids'], chunk['input_ids']), dim=0)
            train_tokens['attention_mask'] = torch.cat((train_tokens['attention_mask'], chunk['attention_mask']), dim=0)

    return train_tokens

class ClassificationDataset(Dataset):
    def __init__(self, id, tokens, stylometric_features):
        super().__init__()
        self.id = id
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.stylometric_features = torch.from_numpy(stylometric_features).float()
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, idx):
        label = self.id[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        stylometric_features = self.stylometric_features[idx]

        return label, input_ids, attention_mask, stylometric_features