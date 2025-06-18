import argparse
import torch
import torch.nn as nn
import pandas as pd
import os
from transformers import AutoModel, AutoTokenizer
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
import numpy as np

from utils.models import ClassificationModel, SiameseModel
from utils.data import tokenize_df, ClassificationDataset
from utils.config import load_config

from transformers.utils import logging

def main():
    # Logging 
    logger = logging.get_logger("transformers")

    parser = argparse.ArgumentParser(description="Train embeddings")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/closed_classification/gte.yaml",
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

    # Load Dataset
    logger.warning("Loading Dataset...")
    
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
    embedding_model = SiameseModel(encoder)

    # Load model weights
    model_path = args.model
    embedding_model.load_state_dict(torch.load(model_path, map_location=device))

    # Freeze the embedding model
    for param in embedding_model.parameters():
        param.requires_grad = False

    model = ClassificationModel(
        embedding_model, 
        num_classes=model_config['num_classes'], 
        num_layers=model_config['num_classification_layers'],
        embedding_size=model_config['final_embedding_dim'])
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
        train_tokens = tokenize_df(tokenizer, train_texts, max_length=train_config['max_length'])
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
        test_tokens = tokenize_df(tokenizer, test_text, max_length=train_config['max_length'])
        torch.save(test_tokens['input_ids'], TEST_CACHE + "_ids.pt")
        torch.save(test_tokens['attention_mask'], TEST_CACHE + "_attention_mask.pt")

    # Load Stylometric, ignore text and id columns
    train_stylometric = train_df.drop(columns=['id', 'text'])
    test_stylometric = test_df.drop(columns=['id', 'text'])
    train_stylometric = train_stylometric.to_numpy()
    test_stylometric = test_stylometric.to_numpy()

    # Map all ids to integers
    ids = pd.concat([train_ids, test_ids]).unique()
    id_to_int = {id_: i for i, id_ in enumerate(ids)}
    train_ids = train_ids.map(id_to_int)
    test_ids = test_ids.map(id_to_int)

    train_dataset = ClassificationDataset(
        id=train_ids.to_numpy(),
        tokens=train_tokens,
        stylometric_features=train_stylometric,
    )

    test_dataset = ClassificationDataset(
        id=test_ids.to_numpy(),
        tokens=test_tokens,
        stylometric_features=test_stylometric,
    )

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_config['batch_size'], 
        shuffle=True, 
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=train_config['batch_size'], 
        shuffle=False, 
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config['initial_lr']))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(train_config['lr_decay']))
    num_epochs = train_config['epochs']

    for epoch in range(num_epochs):

        model.train()

        for step, batch in enumerate(train_loader):
            
            labels, input_ids, attention_mask, stylometric_features = batch

            labels = labels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            stylometric_features = stylometric_features.to(device)

            batch_input = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

            logits = model(batch_input, stylometric_features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if (step+1) % 10 == 0:
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == labels).float().mean().item()
                logger.warning(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_loader)}, Accuracy: {accuracy:.4f}, Loss: {loss.item()}")

            if (step+1) % 100 == 0:
                model.eval()
                preds_list = []
                labels_list = []
                with torch.no_grad():
                    for j, batch in enumerate(test_loader):
                        if j >= 10:
                            break
                        labels, input_ids, attention_mask, stylometric_features = batch

                        labels = labels.to(device)
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        stylometric_features = stylometric_features.to(device)

                        batch_input = {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask
                        }

                        logits = model(batch_input, stylometric_features)
                        preds = torch.argmax(logits, dim=1)
                        preds_list.append(preds.cpu())
                        labels_list.append(labels.cpu())
                avg_acc = multiclass_accuracy(torch.cat(preds_list), torch.cat(labels_list), num_classes=model_config['num_classes'])
                macro_f1 = multiclass_f1_score(torch.cat(preds_list), torch.cat(labels_list), num_classes=model_config['num_classes'], average='macro')
                micro_f1 = multiclass_f1_score(torch.cat(preds_list), torch.cat(labels_list), num_classes=model_config['num_classes'], average='micro')
                logger.warning(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_loader)}, Validation Accuracy: {avg_acc:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")

        # Validation
        model.eval()
        preds_list = []
        labels_list = []
        with torch.no_grad():
            for batch in test_loader:
                labels, input_ids, attention_mask, stylometric_features = batch

                labels = labels.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                stylometric_features = stylometric_features.to(device)

                batch_input = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }

                logits = model(batch_input, stylometric_features)
                preds = torch.argmax(logits, dim=1)
                preds_list.append(preds.cpu())
                labels_list.append(labels.cpu())

        avg_acc = multiclass_accuracy(torch.cat(preds_list), torch.cat(labels_list), num_classes=model_config['num_classes'])
        macro_f1 = multiclass_f1_score(torch.cat(preds_list), torch.cat(labels_list), num_classes=model_config['num_classes'], average='macro')
        micro_f1 = multiclass_f1_score(torch.cat(preds_list), torch.cat(labels_list), num_classes=model_config['num_classes'], average='micro')
        logger.warning(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {avg_acc:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")

        if (epoch+1) % data_config['model_save_freq'] == 0:
            if not os.path.exists(f"{data_config['model_save_dir']}/closed_classification"):
                os.makedirs(f"{data_config['model_save_dir']}/closed_classification")
            torch.save(model.state_dict(), f"{data_config['model_save_dir']}/closed_classification/{data_config['variant']}_{model_config['encoder']}_epoch_{epoch+1}.pt")
            logger.warning(f"Model saved at epoch {epoch+1}")

        scheduler.step()

if __name__ == "__main__":
    main()