import argparse
import torch
import torch.nn.functional as F
from torchpq.index import IVFPQIndex
import pandas as pd
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils.models import SiameseModel
from utils.config import load_config
from utils.data import tokenize_df


from transformers.utils import logging

def main():

    # Logging 
    logger = logging.get_logger("transformers")

    parser = argparse.ArgumentParser(description="Train embeddings")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/ir/gte.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="models/gte.pt",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    train_config = config["training"]
    model_config = config["model"]

    logger.warning(f"Running with the config: {config}")

    if data_config['variant'] == 'all':
        df = pd.read_csv(os.path.join(data_config["root"], f"{data_config['dataset']}_test_{data_config['variant']}.csv"))
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index) 
        TRAIN_CACHE = os.path.join(data_config["root"], f"{data_config['dataset']}_train_{data_config['variant']}_80_{model_config['encoder']}")
        TEST_CACHE = os.path.join(data_config["root"], f"{data_config['dataset']}_test_{data_config['variant']}_20_{model_config['encoder']}")
    else:
        train_df = pd.read_csv(os.path.join(data_config["root"], f"{data_config['dataset']}_train_{data_config['variant']}.csv"))
        test_df = pd.read_csv(os.path.join(data_config["root"], f"{data_config['dataset']}_test_{data_config['variant']}.csv"))
        TRAIN_CACHE = os.path.join(data_config["root"], f"{data_config['dataset']}_train_{data_config['variant']}_{model_config['encoder']}")
        TEST_CACHE = os.path.join(data_config["root"], f"{data_config['dataset']}_test_{data_config['variant']}_{model_config['encoder']}")
    
    # Load Model and Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning("Loading Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased" if model_config['encoder'] == "bert" else "Alibaba-NLP/gte-base-en-v1.5",
        use_fast=True,
        unpad_inputs=True,
        use_memory_efficient_attention=True,
        torch_dtype=torch.float16
    )
    encoder = AutoModel.from_pretrained(
        "distilbert/distilbert-base-uncased" if model_config['encoder'] == "bert" else "Alibaba-NLP/gte-base-en-v1.5",
        trust_remote_code=True, 
    )
    model = SiameseModel(encoder)

    # Load model weights
    model_path = args.model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    logger.warning("Tokenizing Dataset...")
    train_ids = train_df['id']
    train_texts = train_df['text']
    if os.path.exists(TRAIN_CACHE + "_ids.pt") and os.path.exists(TRAIN_CACHE + "_attention_mask.pt"):
        logger.warning("Loading pre-tokenized train dataset...")
        train_tokens = torch.load(TRAIN_CACHE + "_ids.pt")
        train_attention_mask = torch.load(TRAIN_CACHE + "_attention_mask.pt")
        train_tokens = {
            'input_ids': train_tokens,
            'attention_mask': train_attention_mask
        }
    else:
        logger.warning("Tokenizing train dataset...")
        train_tokens = tokenize_df(tokenizer, train_texts)
        torch.save(train_tokens['input_ids'], TRAIN_CACHE + "_ids.pt")
        torch.save(train_tokens['attention_mask'], TRAIN_CACHE + "_attention_mask.pt")

    test_ids = test_df['id']
    test_text = test_df['text']
    if os.path.exists(TEST_CACHE + "_ids.pt") and os.path.exists(TEST_CACHE + "_attention_mask.pt"):
        logger.warning("Loading pre-tokenized test dataset...")
        test_tokens = torch.load(TEST_CACHE + "_ids.pt")
        test_attention_mask = torch.load(TEST_CACHE + "_attention_mask.pt")
        test_tokens = {
            'input_ids': test_tokens,
            'attention_mask': test_attention_mask
        }
    else:
        logger.warning("Tokenizing test dataset...")
        test_tokens = tokenize_df(tokenizer, test_text)
        torch.save(test_tokens['input_ids'], TEST_CACHE + "_ids.pt")
        torch.save(test_tokens['attention_mask'], TEST_CACHE + "_attention_mask.pt")

    # Load Stylometric, ignore text and id columns
    train_stylometric = train_df.drop(columns=['id', 'text'])
    test_stylometric = test_df.drop(columns=['id', 'text'])
    train_stylometric = train_stylometric.to_numpy()
    test_stylometric = test_stylometric.to_numpy()

    train_stylometric = torch.tensor(train_stylometric).float()
    test_stylometric = torch.tensor(test_stylometric).float()

    # Calculate embeddings for train set
    logger.warning("Calculating train embeddings...")

    if os.path.exists(TRAIN_CACHE + "_embeddings.pt"):
        logger.warning("Loading pre-computed train embeddings...")
        train_embeddings = torch.load(TRAIN_CACHE + "_embeddings.pt")
    else:
        train_embeddings = torch.zeros((len(train_tokens['input_ids']), model_config['final_embedding_dim']))
        with torch.no_grad():
            for i in tqdm(range(0, len(train_tokens['input_ids']), train_config['batch_size'])):
                batch_ids = train_tokens['input_ids'][i:i+train_config['batch_size']].to(device)
                batch_attention_mask = train_tokens['attention_mask'][i:i+train_config['batch_size']].to(device)
                batch_stylometric = train_stylometric[i:i+train_config['batch_size']].to(device)
                batch_input = {
                    'input_ids': batch_ids,
                    'attention_mask': batch_attention_mask
                }

                batch_embeddings = model(batch_input, batch_stylometric)
                train_embeddings[i:i+train_config['batch_size']] = batch_embeddings.cpu()

        # Save
        if not os.path.exists("data"):
            os.makedirs("data")
        torch.save(train_embeddings, TRAIN_CACHE + "_embeddings.pt")

    # Get test embeedings
    logger.warning("Calculating test embeddings...")
    if os.path.exists(TEST_CACHE + "_embeddings.pt"):
        logger.warning("Loading pre-computed test embeddings...")
        test_embeddings = torch.load(TEST_CACHE + "_embeddings.pt")
    else:
        test_embeddings = torch.zeros((len(test_tokens['input_ids']), model_config['final_embedding_dim']))
        with torch.no_grad():
            for i in tqdm(range(0, len(test_tokens['input_ids']), train_config['batch_size'])):
                batch_ids = test_tokens['input_ids'][i:i+train_config['batch_size']].to(device)
                batch_attention_mask = test_tokens['attention_mask'][i:i+train_config['batch_size']].to(device)
                batch_stylometric = test_stylometric[i:i+train_config['batch_size']].to(device)
                batch_input = {
                    'input_ids': batch_ids,
                    'attention_mask': batch_attention_mask
                }

                batch_embeddings = model(batch_input, batch_stylometric)
                test_embeddings[i:i+train_config['batch_size']] = batch_embeddings.cpu()

        # Save
        if not os.path.exists("data"):
            os.makedirs("data")
        torch.save(test_embeddings, TEST_CACHE + "_embeddings.pt")

    # Create IVFPQ index
    index = IVFPQIndex(
        d_vector=model_config['final_embedding_dim'],
        n_subvectors=32,
        n_cells=4096,
        initial_size=2048,
        distance="cosine",
    )

    train_embeddings = train_embeddings.T.contiguous()
    train_embeddings = train_embeddings.to(device)
    train_embeddings = F.normalize(train_embeddings, p=2, dim=1)
    train_embeddings = train_embeddings.float()

    index.train(train_embeddings)
    index.add(train_embeddings)

    test_embeddings = test_embeddings.T.contiguous()
    test_embeddings = test_embeddings.to(device)
    test_embeddings = F.normalize(test_embeddings, p=2, dim=1)
    test_embeddings = test_embeddings.float()

    logger.warning("Querying the index for test embeddings...")
    # Get count of each author in the train set
    author_counts = train_df['id'].value_counts().to_dict()
    train_ids = torch.tensor(train_ids.to_numpy(), dtype=torch.int64, device=device)
    test_ids = torch.tensor(test_ids.to_numpy(), dtype=torch.int64)

    precisions = []
    mrrs = []

    k = 16
    index.n_probe = 32
    _, topk_ids = index.search(test_embeddings, k=k)

    for i in range(len(test_ids)):
        author_id = test_ids[i]
        if author_id.item() not in author_counts:
            continue
        author_count = author_counts[author_id.item()]

        # Get the top-k results for the test embedding
        author_count = min(author_count, k)
        retrieved_authors = train_ids[topk_ids[i, :author_count]]

        relevant = (retrieved_authors == author_id).sum().item()
        precision = relevant / len(retrieved_authors)

        rank = (retrieved_authors == author_id).nonzero(as_tuple=True)[0]
        rank = rank[0] if len(rank) > 0 else len(retrieved_authors)
        rank = 1 / (rank+1)

        precisions.append(precision)
        mrrs.append(rank)

    avg_precision = sum(precisions) / len(precisions)
    avg_mrr = sum(mrrs) / len(mrrs)
    logger.warning(f"Retrieval Precision: {avg_precision:.4f}")
    logger.warning(f"Retrieval MRR: {avg_mrr:.4f}")

if __name__ == "__main__":
    main()