import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging

from utils.config import load_config
from utils.data import tokenize_df, NCEDataset
from utils.loss import (
    cosine_similarity,
    MultipleNegativesSymmetricRankingLoss,
)
from utils.metrics import f1
from utils.models import SiameseModel

def main():
    # Logging 
    logger = logging.get_logger("transformers")

    parser = argparse.ArgumentParser(description="Train embeddings")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/embeddings/gte.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    train_config = config["training"]
    model_config = config["model"]

    logger.warning(f"Running with the config: {config}")

    # Load Dataset
    logger.warning("Loading Dataset...")
    
    train_df = pd.read_csv(os.path.join(data_config["root"], f"{data_config['dataset']}_train_{data_config['variant']}.csv"))
    test_df = pd.read_csv(os.path.join(data_config["root"], f"{data_config['dataset']}_test_{data_config['variant']}.csv"))
    TRAIN_CACHE = os.path.join(data_config["root"], f"{data_config['dataset']}_test_{data_config['variant']}_{model_config['encoder']}")
    TEST_CACHE = os.path.join(data_config["root"], f"{data_config['dataset']}_test_{data_config['variant']}_{model_config['encoder']}")

    # Load Model and Tokenizer
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

    # Freeze the model parameters
    for param in encoder.parameters():
        param.requires_grad = False

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

    logger.warning("Creating Dataset and DataLoader...")

    # Load Stylometric, ignore text and id columns
    train_stylometric = train_df.drop(columns=['id', 'text'])
    test_stylometric = test_df.drop(columns=['id', 'text'])
    train_stylometric = train_stylometric.to_numpy()
    test_stylometric = test_stylometric.to_numpy()

    # Create Dataset and DataLoader
    train_dataset = NCEDataset(train_ids.to_numpy(), train_tokens, train_stylometric)
    test_dataset = NCEDataset(test_ids.to_numpy(), test_tokens, test_stylometric)

    train_dataloader = DataLoader(train_dataset, batch_size=train_config['train_batch'], shuffle=True, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=train_config['test_batch'], shuffle=False, pin_memory=True)

    # Create Model
    model = SiameseModel(encoder, pooling=model_config['encoder_pooling'], embedding_size=model_config['final_embedding_dim'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning("Using device: " + str(device))
    model.to(device)

    # Loss Function and Optimizer
    criterion = MultipleNegativesSymmetricRankingLoss(scale=float(train_config['scale']))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config['initial_lr']))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(train_config['lr_decay']))

    epochs = train_config['epochs']
    mini_test_size = 0.005 * len(train_dataloader)
    frozen = True
    freeze_until = train_config['freeze_until']

    logger.warning("Training Model...")
    # Training Loop
    for epoch in range(epochs):

        # Start fine-tuning encoder
        if frozen and (epoch+1) >= freeze_until:
            logger.warning("Unfreezing encoder parameters...")
            for param in model.parameters():
                param.requires_grad = True
            frozen = False
            for g in optimizer.param_groups:
                g['lr'] = float(train_config['unfrozen_lr'])
            train_dataloader = DataLoader(train_dataset, batch_size=train_config['unfrozen_train_batch'], shuffle=True, pin_memory=True, drop_last=True)

        model.train()
        for i, batch in enumerate(train_dataloader):

            start_time = time.time()

            a_input_ids = batch['anchor_input_ids']
            p_input_ids = batch['positive_input_ids']
            a_attention_mask = batch['anchor_attention_mask']
            p_attention_mask = batch['positive_attention_mask']
            a_styolometric_features = batch['anchor_styolometric_features']
            p_styolometric_features = batch['positive_styolometric_features']

            a_input_ids = a_input_ids.to(device)
            p_input_ids = p_input_ids.to(device)
            a_attention_mask = a_attention_mask.to(device)
            p_attention_mask = p_attention_mask.to(device)
            a_styolometric_features = a_styolometric_features.to(device)
            p_styolometric_features = p_styolometric_features.to(device)

            # Forward pass
            anchor = {
                'input_ids': a_input_ids,
                'attention_mask': a_attention_mask
            }
            positive = {
                'input_ids': p_input_ids,
                'attention_mask': p_attention_mask
            }

            # Forward pass
            anchor_output = model(anchor, a_styolometric_features)
            positive_output = model(positive, p_styolometric_features)

            loss = criterion(anchor_output, positive_output)

            # Backward pass
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            
            if (i+1) % 50 == 0:
                dists = cosine_similarity(anchor_output, positive_output)
                pos_dist = dists.diagonal()
                neg_dist = dists.flatten()[1:].view(dists.shape[0]-1, dists.shape[0]+1)[:,:-1].flatten()

                preds = dists.argmax(dim=1) == torch.arange(dists.shape[0]).to(device)
                labels = torch.ones_like(preds)

                # Calculate F1 score
                f1_score = f1(preds, labels)

                acc = (preds == labels).float().mean()

                # Compute MRR
                rank = dists.argsort(dim=1,descending=True, stable=True).argsort(dim=1).diagonal()
                rank = 1 / (rank+1)
                mrr = torch.mean(rank)

                logger.warning(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, F1: {f1_score:.4f}, Acc: {acc:.4f}, MRR {mrr:.4f} , Time: {time.time() - start_time:.2f}s")
                logger.warning(f"Mean Positive Distance {str(torch.mean(pos_dist).cpu().item())}")
                logger.warning(f"Mean Negative Distance {str(torch.mean(neg_dist).cpu().item())}")

        scheduler.step()
    
        # Evaluation Loop
        if epoch % 5 == 0:
            logger.warning("Peforming full evaluation...")
            j = len(test_dataloader)
        else:
            j = mini_test_size
        model.eval()
        f1_scores = []
        acc_scores = []
        mrr_scores = []
        pos_dists = []
        neg_dists = []
        with torch.no_grad():
            start_time = time.time()
            for j, batch in enumerate(test_dataloader):
                if j >= mini_test_size:
                    break

                a_input_ids = batch['anchor_input_ids']
                p_input_ids = batch['positive_input_ids']
                a_attention_mask = batch['anchor_attention_mask']
                p_attention_mask = batch['positive_attention_mask']
                a_styolometric_features = batch['anchor_styolometric_features']
                p_styolometric_features = batch['positive_styolometric_features']

                a_input_ids = a_input_ids.to(device)
                p_input_ids = p_input_ids.to(device)
                a_attention_mask = a_attention_mask.to(device)
                p_attention_mask = p_attention_mask.to(device)
                a_styolometric_features = a_styolometric_features.to(device)
                p_styolometric_features = p_styolometric_features.to(device)

                # Forward pass
                anchor = {
                    'input_ids': a_input_ids,
                    'attention_mask': a_attention_mask
                }
                positive = {
                    'input_ids': p_input_ids,
                    'attention_mask': p_attention_mask
                }

                # Forward pass
                anchor_output = model(anchor, a_styolometric_features)
                positive_output = model(positive, p_styolometric_features)

                dists = cosine_similarity(anchor_output, positive_output)
                pos_dist = dists.diagonal()
                neg_dist = dists.flatten()[1:].view(dists.shape[0]-1, dists.shape[0]+1)[:,:-1].flatten()
                preds = dists.argmax(dim=1) == torch.arange(dists.shape[0]).to(device)
                labels = torch.ones_like(preds)

                # Calculate F1 score
                f1_score = f1(preds, labels)

                acc = (preds == labels).float().mean()

                # Compute MRR
                rank = dists.argsort(dim=1,descending=True, stable=True).argsort(dim=1).diagonal()
                rank = 1 / (rank+1)
                mrr = torch.mean(rank)

                f1_scores.append(f1_score)
                acc_scores.append(acc.cpu().item())
                mrr_scores.append(mrr.cpu().item())
                pos_dists.append(pos_dist)
                neg_dists.append(neg_dist)

            end_time = time.time()
            logger.warning(f"Test F1 Score: {np.mean(f1_scores):.4f}")
            logger.warning(f"Test Acc: {np.mean(acc_scores):.4f}")
            logger.warning(f"Test MRR: {np.mean(mrr_scores):.4f}")
            logger.warning(f"Test Pos Dist: {torch.median(torch.cat(pos_dists)):.4f}")
            logger.warning(f"Test Neg Dist: {torch.median(torch.cat(neg_dists)):.4f}")
            logger.warning(f"Test Time per step: {(end_time - start_time) / j:.2f}s")

        model.train()

        # Save Model
        if (epoch+1) % data_config['model_save_freq']:
            if not os.path.exists(f"{data_config['model_save_dir']}/embedding"):
                os.makedirs(f"{data_config['model_save_dir']}/embedding")
            torch.save(model.state_dict(), f"{data_config['model_save_dir']}/embedding/{data_config['variant']}_{model_config['encoder']}_epoch_{epoch+1}.pt")

    if not os.path.exists(f"{data_config['model_save_dir']}/embedding"):
            os.makedirs(f"{data_config['model_save_dir']}/embedding")
    torch.save(model.state_dict(), f"{data_config['model_save_dir']}/embedding/{data_config['variant']}_{model_config['encoder']}_final.pt")

if __name__ == "__main__":
    main()


