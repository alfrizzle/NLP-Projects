"""
This tutorial borrows from https://huggingface.co/transformers/custom_datasets.html#seq-imdb for
"""
import argparse
import os
import json
import pathlib
import logging
import sys

# import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from utils import read_object

# Set up logging
logger = logging.getLogger(__name__)

logging.basicConfig(
  level=logging.getLevelName("INFO"),
  handlers=[logging.StreamHandler(sys.stdout)],
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  )

class ClassifierDataset(Dataset):
    """Inherits from torch.utils.data.Dataset.
    Arguments:
        data (pd.DataFrame): Contains columns 'input' and 'label'.
        model_name (str): Name of the transformers model to load the tokenizer.
        max_sequence_length (int): Maximum length of the sequence.
    """
    def __init__(self, data, model_name, max_sequence_length):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_sequence_length = max_sequence_length

    def __getitem__(self, index):
        """Returns the model inputs at the specified index."""
        text_input = self.data["input"].iloc[index]
        input_dict = self.tokenizer.encode_plus(
            text_input,
            max_length=args.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {
            "input_ids": input_dict["input_ids"][0],
            "attention_mask": input_dict["attention_mask"][0],
            "labels": torch.tensor(self.data["label"].iloc[index], dtype=torch.long)
        }
        return inputs

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)

def preprocess_data(df):
    """Pre-process data for training."""
    # # Convert 5 star scale to binary -- positive (1) or negative (0).
    # star_map = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    # df["label"] = df["star_rating"].map(star_map)
    # # Concatenate the review headline with the body for model input.
    # df["input"] = df["review_headline"] + " " + df["review_body"]
    # # Drop nans.
    # df = df.loc[df["input"].notnull()].copy()
    # # Split data into train and valid.
    # traindf, validdf = train_test_split(df[["input", "label"]])
    # df = read_object(df, file_type)
    traindf, testdf = train_test_split(df, test_size=args.test_size, stratify=df['label'])
    return traindf, testdf

def compute_metrics(output):
    """Compute custom metrics to track in logging."""
    y_true = output.label_ids
    y_pred = output.predictions.argmax(-1)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    # auprc = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred, average=None)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

def main(args):
    """Executes training job."""
    # Load data into a pandas DataFrame.
    df = read_object(args.input_path, args.file_type)
    if args.max_data_rows:
        df = df.iloc[:args.max_data_rows].copy()
    logger.info(f" Dataset contains {len(df)} rows")
    print(f"Data contains {len(df)} rows")

    # Preprocess and featurize data.
    traindf, validdf = preprocess_data(df)

    # Create data set for model input.
    train_dataset = ClassifierDataset(traindf, args.model_name, args.max_sequence_length)
    valid_dataset = ClassifierDataset(validdf, args.model_name, args.max_sequence_length)
    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(valid_dataset)}")
    print(train_dataset[0])

    # Define model and trainer.
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     args.model_name,
    #     num_labels=2,
    #     torchscript=True
    # )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.model_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=True
    )

    # create trainer instance
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model.
    if get_last_checkpoint(args.model_dir) is not None:
        logger.info("***** continue training *****")
        last_checkpoint = get_last_checkpoint(args.model_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # Evaluate the model
    # eval = trainer.evaluate()
    eval_result = trainer.evaluate(eval_dataset=valid_dataset)
    
    # Select only the 5 relevant metrics keys
    keep = ['eval_f1_score', 'eval_loss', 'eval_precision', 'eval_recall', 'eval_roc_auc']
    partial_dict = {k: v for k, v in eval_result.items() if k in keep}

    # initializing new dict key names
    dict_key_map = {'eval_loss' : 'Loss',
                    'eval_precision' : 'Precision',
                    'eval_recall'    : 'Recall', 
                    'eval_f1_score'  : 'f1',
                    'eval_roc_auc'   : 'ROC_AUC'
                    }
    
    # changing keys of dictionary
    final_dict = {dict_key_map.get(k, k): v for k, v in partial_dict.items()}

    # Save eval results as json file in new evaluation folder
    # pathlib.Path("evaluation").mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.eval_dir, 'eval_results.json'), 'w') as fp:
      print('****** Eval Results ******')
      json.dump(final_dict, fp)

    # Save the trained model
    trainer.save_model(args.model_dir)
    # trainer.save_state()

    # # Save the model to the output_path defined in train_model.py.
    # device = torch.device("cuda")
    # dummy_row = train_dataset[0]
    # dummy_input = (
    #     dummy_row["input_ids"].unsqueeze(0).to(device),
    #     dummy_row["attention_mask"].unsqueeze(0).to(device)
    # )
    # traced_model = torch.jit.trace(model, dummy_input)
    # torch.jit.save(traced_model, os.path.join(args.model_dir, "model.pth"))


if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters from launch_training_job.py get passed in as command line args.
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--test_size', type=float, default=.2)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_data_rows', type=int, default=None)
    parser.add_argument('--max_sequence_length', type=int, default=128)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=128)
    parser.add_argument('--file_type', type=str, default='csv') # specify whether input file is csv or json (has to be one of the two)
    parser.add_argument('--eval_dir', type=str, default='../model/evaluation') # set this to SM's model_dir path when using in SageMaker
    parser.add_argument('--model_dir', type=str, default='../model') # where trained model is saved when running the script locally (outside of SageMaker)
    
    # # SageMaker environment variables.
    # parser.add_argument('--hosts', type=list, default=os.environ['SM_HOSTS'])
    # parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    # parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR']) # output_path arg from train_model.py.
    # parser.add_argument('--num_cpus', type=int, default=os.environ['SM_NUM_CPUS'])
    # parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    # Parse command-line args and run main.
    args = parser.parse_args()
    main(args)
