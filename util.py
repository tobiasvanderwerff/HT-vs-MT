import argparse
from collections import defaultdict
from typing import Optional, Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import TrainingArguments, EvalPrediction


class HFDataset(torch.utils.data.Dataset):
    """Dataset for using HuggingFace Transformers."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(
    pred: EvalPrediction, idx_to_docid: Optional[Dict[int, int]] = None
):
    labels = pred.label_ids
    # Sometimes the output is a tuple, take first argument then.
    if isinstance(pred.predictions, tuple):
        pred = pred.predictions[0]
    else:
        pred = pred.predictions
    preds = pred.argmax(-1)
    if idx_to_docid is not None:
        # Majority voting: take the most common prediction per document.
        assert len(idx_to_docid) == len(preds), f"{len(idx_to_docid)} vs {len(preds)}"
        docid_to_preds = defaultdict(list)
        docid_to_label = dict()
        for idx, (p, l) in enumerate(zip(preds, labels)):
            docid = idx_to_docid[idx]
            docid_to_preds[docid].append(p)
            docid_to_label[docid] = l
        preds_new = []
        for docid, doc_preds in docid_to_preds.items():
            # Take the majority prediction.
            perc = sum(doc_preds) / len(doc_preds)
            preds_new.append(1 if perc >= 0.5 else 0)
        preds = np.array(preds_new)
        labels = np.array(list(docid_to_label.values()))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def get_training_arguments(args):
    """Load all training arguments here. There are a lot more not specified, check:
    https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py#L72"""
    return TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.strategy,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.use_fp16,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        logging_strategy=args.strategy,
        logging_steps=args.logging_steps,
        save_strategy=args.strategy,
        save_steps=args.save_steps,
        seed=args.seed,
        load_best_model_at_end=True,
        label_smoothing_factor=args.label_smoothing,
        log_level="debug",
        metric_for_best_model="accuracy",
        save_total_limit=2,
    )


def parse_args_hf():
    """
    Parse CLI arguments for the script and return them.
    :return: Namespace of parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Arguments for running the classifier."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./experiments/8",
        help="Location of the root directory. By default, this is "
        "the data from WMT08-19, without Translationese.",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        help="Initialize training from the model specified at this " "path location.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        help=("Huggingface transformer architecture to use, " "e.g. `bert-base-cased`"),
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "-wd", "--weight_decay", default=0, type=float, help="Weight decay"
    )
    parser.add_argument(
        "-mgn", "--max_grad_norm", default=1, type=float, help="Max grad norm"
    )
    # parser.add_argument("-wr", "--warmup_ratio", default=0.1, type=float,
    #                     help="Ratio of total training steps used for a linear warmup "
    #                          "from 0 to learning_rate.")
    parser.add_argument(
        "-wr",
        "--warmup_steps",
        default=200,
        type=int,
        help="Number of steps used for a linear warmup from 0 to " "learning_rate",
    )
    parser.add_argument(
        "-ls",
        "--label_smoothing",
        default=0.0,
        type=float,
        help="Label smoothing percentage, 0-1",
    )
    parser.add_argument(
        "-dr",
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout applied to the classifier layer",
    )
    parser.add_argument(
        "-str",
        "--strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default="steps",
        help="Strategy for evaluating/saving/logging",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Number of update steps between two evaluations",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=200,
        help="Number of update steps between two logs",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Number of update steps before two checkpoints saves",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--early_stopping_patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--use_fp16", action="store_true", help="Use mixed 16-bit precision"
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1e30)
    parser.add_argument(
        "--load_sentence_pairs",
        action="store_true",
        help="Set this flag to classify HT vs. MT for "
        "source/translation pairs, rather than just "
        "translations.",
    )
    parser.add_argument(
        "--use_google_data",
        action="store_true",
        help="Use Google Translate data instead of DeepL data for train/dev/test.",
    )
    parser.add_argument(
        "--use_normalized_data",
        action="store_true",
        help="Use translations that have been post-processed by applying "
        "a Moses normalization script to them. Right now only works for "
        "monolingual sentences",
    )
    parser.add_argument(
        "--use_majority_classification",
        action="store_true",
        help="Make predictions by predicting each segment in a "
        "document and taking the majority prediction. This is "
        "only used for evaluating an already trained "
        "sentence-level model on documents.",
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["deepl", "google", "wmt1", "wmt2", "wmt3", "wmt4"],
        help="Test a classifier on one of the test sets. For WMT "
        "submissions there are 4 options, originating from the "
        "WMT 19 test set. Along with their DA scores:"
        "- wmt1: Facebook-FAIR (best, 81.6)"
        "- wmt2: RWTH-Aachen (2nd best, 81.5)"
        "- wmt3: PROMPT-NMT (2nd worst, 71.8)"
        "- wmt4: online-X (worst, 69.7)",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate on dev set using a trained model"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random number generator seed."
    )
    return parser.parse_args()
