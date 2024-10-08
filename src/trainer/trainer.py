# %% [markdown]
# <a href="https://colab.research.google.com/github/gianclbal/ALMA-TACIT/blob/main/data-analysis/exp_redos/exp_1/exp_1_atn_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
# Importing the libraries needed
import argparse
import json
import logging
import os
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# %%
def rename_and_encode(df):
    # Rename the 'sentences' column to 'sentence'
    df = df.rename(columns={'sentences': 'sentence'})

    # Check if 'labels' column already contains 0 and 1 values
    unique_values = df['label'].unique()

    if set(unique_values) == {0, 1}:
        # If the unique values are 0 and 1, no encoding is needed
        return df
    else:
        # Otherwise, map 'Yes' to 1 and 'No' to 0 in the 'labels' column
        df['label'] = df['label'].map({'Yes': 1, 'No': 0})

    return df

class SentenceData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, model):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe["sentence"]
        self.targets = self.data["label"]
        self.max_len = max_len
        self.model = model

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        if self.model == "roberta-base" or self.model == "bert-base-uncased":
            token_type_ids = inputs["token_type_ids"]
        else:
            token_type_ids = None

        return_dict = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

        if token_type_ids is not None:
            return_dict['token_type_ids'] = torch.tensor(token_type_ids, dtype=torch.float)

        return return_dict

def data_loader(train_df, test_df, max_len, list_of_model_names):

    datasets = {}

    X = train_df['sentence']
    y = train_df['label']

    # Split the data
    train_dataset, validation_dataset = train_test_split(train_df, test_size=0.1, random_state=18, stratify=training_df.label)

    train_dataset.reset_index(drop=True, inplace=True)
    validation_dataset.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VALIDATION Dataset: {}".format(validation_dataset.shape))
    print("TEST Dataset: {}".format(test_df.shape))

    # data loader parameters
    train_params = {'batch_size': BATCH_SIZE,
                # 'shuffle': True,
                'num_workers': 0
                }

    validate_params = {'batch_size': BATCH_SIZE,
                    # 'shuffle': True,
                    'num_workers': 0
                    }
    test_params = {'batch_size': BATCH_SIZE,
                    # 'shuffle': True,
                    'num_workers': 0
                    }

    for model_name in list_of_model_names:
        training_set = SentenceData(train_dataset, AutoTokenizer.from_pretrained(model_name), max_len, model_name)
        validate_set = SentenceData(validation_dataset, AutoTokenizer.from_pretrained(model_name), max_len, model_name)
        testing_set = SentenceData(test_df, AutoTokenizer.from_pretrained(model_name), max_len, model_name)

        training_loader = DataLoader(training_set, **train_params)
        validate_loader = DataLoader(validate_set, **validate_params)
        testing_loader = DataLoader(testing_set, **test_params)

        datasets[model_name] = {'train': training_loader, 'test': testing_loader, 'validate': validate_loader}

    return datasets

class TextClassificationModel(torch.nn.Module):

    def __init__(self, model_name, num_classes):
        super(TextClassificationModel, self).__init__()

        # Load the pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Ensure compatibility with different model types
        if "roberta" in model_name or "bert" in model_name:
            self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_classes)
        else:
            # Handle models that might have different output dimensions
            raise NotImplementedError(f"Model {model_name} not currently supported")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Handle token_type_ids based on model type
        if token_type_ids is not None and "distilbert" not in self.model.config.model_type:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # if model_name != "distilbert-base-uncased":
        #     # Access logits directly for sequence classification models
        logits = outputs.logits
        # else:
        #     # DistilBERT doesn't have a pooler output, so directly use last hidden state
        #     pooled_output = outputs.last_hidden_state[:, 0]  # Extract the first token's hidden state
        #     # Pass the pooled output through the classifier layer to obtain the logits
        #     logits = self.classifier(pooled_output)

        # Apply softmax activation to obtain the probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)


        return probs

import torch
from torch.optim.lr_scheduler import LambdaLR

class TriangularLR(LambdaLR):
    def __init__(self, optimizer, base_lr, max_lr, num_training_steps, num_warmup_steps, last_epoch=-1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps

        def lr_lambda(current_step):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + progress) * (1.0 - progress))

        super().__init__(optimizer, lr_lambda, last_epoch)

    def get_last_lr(self):
        return [base_lr + (max_lr - base_lr) * max(0, 0.5 * (1.0 + self.last_epoch / float(self.num_training_steps)) * (1.0 - 1.0 / float(self.num_training_steps))) for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)]


def train_model(train_loader, val_loader, model_name, model, epochs, optimizer, scheduler):

    torch.random.manual_seed(18)
    print("Started training on ", device)

    model.to(device)

    class_weights = torch.FloatTensor(weights).to(device) # Assuming two classes, with class 1 having a weight of 5
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    report = {
        'model_name': model_name,
        'train_loss_history': [],
        'val_loss_history': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'macro_p_r_f1_scores': None,
        'roc_auc_score': None,
        'val_confusion_matrix': None
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        train_targets = []
        train_predictions = []

        progress_bar = tqdm(enumerate(train_loader, 0), desc=f'Epoch {epoch+1}/{epochs}, Training', leave=False)
        for _, data in progress_bar:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            if 'token_type_ids' in data:
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            targets = data['targets'].to(device, dtype=torch.long)
            train_targets.extend(targets.tolist())

            optimizer.zero_grad()

            if model_name != "distilbert-base-uncased" and 'token_type_ids' in data:
                outputs = model(ids, mask, token_type_ids)
            else:
                outputs = model(ids, mask)

            _, predicted = torch.max(outputs, 1)
            train_predictions.extend(predicted.tolist())

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


            train_loss += loss.item() * ids.size(0)
            correct_train += (predicted == targets).sum().item()
            total_train += targets.size(0)

            progress_bar.set_postfix({'loss': loss.item()})

            if scheduler != None:
                scheduler.step()

        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        report['train_loss_history'].append(train_loss)
        report['train_accuracy'].append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_targets = []
        val_predictions = []

        progress_bar = tqdm(enumerate(val_loader, 0), desc=f'Epoch {epoch+1}/{epochs}, Validating', leave=False)
        with torch.no_grad():
            for _, data in progress_bar:
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)

                if 'token_type_ids' in data:
                    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

                targets = data['targets'].to(device, dtype=torch.long)
                val_targets.extend(targets.tolist())

                if model_name != "distilbert-base-uncased" and 'token_type_ids' in data:
                    outputs = model(ids, mask, token_type_ids)
                else:
                    outputs = model(ids, mask)

                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.tolist())

                loss = criterion(outputs, targets)
                val_loss += loss.item() * ids.size(0)

                correct_val += (predicted == targets).sum().item()
                total_val += targets.size(0)

                progress_bar.set_postfix({'loss': loss.item()})

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct_val / total_val
        report['val_loss_history'].append(val_loss)
        report['val_accuracy'].append(val_accuracy)

        # Compute classification report and confusion matrix
        report['macro_p_r_f1_scores'] = precision_recall_fscore_support(val_targets, val_predictions, average='macro')
        report['roc_auc_score'] = roc_auc_score(val_targets, val_predictions, average='macro')
        report['val_confusion_matrix'] = confusion_matrix(val_targets, val_predictions)

        # Print epoch results
        # print(f'Epoch [{epoch+1}/{epochs}], '
        #       f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
        #       f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    return model, report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("training_df_path", type=str, help="Path to the training dataset")
    parser.add_argument("test_df_path", type=str, help="Path to the test dataset")
    parser.add_argument("output_name", type=str, help="Dataset name for output name")
    parser.add_argument("--epochs", nargs="+", type=int, default=[10], help="List of epochs")

    args = parser.parse_args()
    
    training_path = args.training_df_path
    test_path = args.test_df_path
    given_epochs = args.epochs
    output_name = args.output_name

    BATCH_SIZE = 6
    WEIGHT_DECAY = 0.01

    print(f"Training dataset path: {args.training_df_path}")
    print(f"Test dataset path: {args.test_df_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Epochs: {args.output_name}")

    training_df = pd.read_csv(training_path, encoding='utf-8')
    print("training set shape", training_df.shape)
    test_df = pd.read_csv(test_path,encoding='utf-8')
    print("test set shape", test_df.shape)
    # augmented_data = pd.read_csv("../../new_data/attainment/augmented_dataset/atn_augmented_dataset_1155.csv")
    print("Training and test sets loaded.")

    training_df = rename_and_encode(training_df)
    test_df = rename_and_encode(test_df)


    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(device)
    
    y = training_df["label"]
        # Calculate class weights
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    weights

    MAX_LEN = 150
    list_of_model_names = ['roberta-base','bert-base-uncased', 'distilbert-base-uncased']

    
    exp_2_datasets = data_loader(train_df=training_df,
                test_df=test_df,
                max_len=MAX_LEN,
            list_of_model_names=list_of_model_names)
    
    # OFFICIAL HYPERPARAM
    # list_of_model_names = ['roberta-base','bert-base-uncased', 'distilbert-base-uncased']
    # learing_rate_list = [2e-5, 0.00001] 
    # scheduler_list = ['cosine','triangular', 'constant']
    # epoch_list = given_epochs

    # JUST ROBERTA WITH BEST HYPERPARAMS FOR LR AND SCHED
    list_of_model_names = ['roberta-base']
    learing_rate_list = [2e-5, 2e-6, 2e-7] 
    scheduler_list = ['triangular']
    epoch_list = given_epochs

    # DUMMY HYPERPARAM TO TEST
    # list_of_model_names = ['distilbert-base-uncased'] #,'bert-base-uncased', 'distilbert-base-uncased'
    # learing_rate_list = [2e-5] 
    # scheduler_list = ['triangular'] #
    # epoch_list = given_epochs
    print('epoch list', epoch_list)


    best_macro_f1 = -1  # Initialize the best macro F1 score
    best_model_info = {}  # Initialize a dictionary to store information about the best model

    results = []

    for model_name in list_of_model_names:
        for lr_option in learing_rate_list:
            for scheduler_option in scheduler_list:
                for epochs in epoch_list:
                    # print(f"Model Name: {model_name}, Learning Rate: {lr_option}, Scheduler Params: {scheduler_option}")
                    # Initialize model, optimizer, and scheduler with current hyperparameters
                    train_loader = exp_2_datasets[model_name]["train"]
                    val_loader = exp_2_datasets[model_name]['validate']
                    model = TextClassificationModel(model_name, num_classes=2)
                
                    optimizer = optim.Adam(model.parameters(), lr=lr_option)
                    
                    if scheduler_option == 'cosine':
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
                    elif scheduler_option == 'triangular':
                        total_samples_in_train = len(train_loader.dataset.data) # it doesn't matter model name
                        total_batches_per_epoch = total_samples_in_train // BATCH_SIZE
                        num_training_steps = total_batches_per_epoch * epochs
                        num_warmup_steps = 0.1 * num_training_steps
                        scheduler = TriangularLR(optimizer, base_lr=lr_option/10, max_lr=lr_option, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
                    elif scheduler_option == 'constant':
                        scheduler = None

                    # Train the model with current hyperparameters
                    trained_model, report = train_model(train_loader, val_loader, model_name, model, epochs, optimizer, scheduler)

                    train_loss_report = report['train_loss_history'][-1]
                    val_loss_report = report['val_loss_history'][-1]
                    macro_scores = report['macro_p_r_f1_scores']
                    auc_score = report['roc_auc_score']

                    # Append to results list
                    results.append({
                        'Model Name': model_name,
                        'Learning Rate': lr_option,
                        'Scheduler Params': scheduler_option,
                        'Epochs': epochs,
                        'Train Loss': train_loss_report,
                        'Validation Loss': val_loss_report,
                        'Macro Precision': macro_scores[0],
                        'Macro Recall': macro_scores[1],
                        'Macro F1': macro_scores[2],
                        'AUC': auc_score
                    })

                    print("*"*20)
                    print(f"Model Name: {model_name}, Learning Rate: {lr_option}, Scheduler Params: {scheduler_option}, Epochs: {epochs}, Train Loss: {train_loss_report}, Validation Loss: {val_loss_report}, Macro Precision: {macro_scores[0]}, Macro Recall: {macro_scores[1]}, Macro F1: {macro_scores[2]}, AUC: {auc_score}")

                    print("*"*20)
                    # Get the final validation loss
                    final_val_loss = report['val_loss_history'][-1]

                    # Check if the final validation loss is better than the current best loss
                    if macro_scores[2] > best_macro_f1:
                        best_macro_f1 = macro_scores[2]
                        best_hyperparameters = {'model_name': model_name, 'epochs': epochs, 'learning_rate': lr_option, 'scheduler_params': scheduler_option}

    # Print the best f1 score and the corresponding hyperparameters
    print("Best Macro F1 score:", best_macro_f1)
    print("Best Hyperparameters:", best_hyperparameters)

    result_df = pd.DataFrame(results)
    output_path = "/data1/gian/sample_dataset/results"
    result_df.to_csv(output_path + f"/{output_name}_gridsearch.csv", index=False)