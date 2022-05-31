from functools import partial
import sys
from pathlib import Path

import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    EarlyStoppingCallback,
    XLMRobertaForSequenceClassification,
    XLMRobertaModel,
)

from data import load_corpus, load_corpus_sentence_pairs
from models import BilingualSentenceClassifier
from util import get_training_arguments, compute_metrics, parse_args_hf


def main():
    """
    Train a model using the Huggingface Trainer API.
    """
    # Get arguments.
    args = parse_args_hf()

    # Set random seed.
    np.random.seed(args.seed)

    # Set directories.
    root_dir = Path(args.root_dir)

    if args.load_model is not None:  # initialize a trained model
        assert Path(
            args.load_model
        ).is_dir(), (
            f"{args.load_model} is not a checkpoint directory, which it should be."
        )

    model_name = args.arch.replace("/", "-")
    mt = "google" if args.use_google_data else "deepl"
    eff_bsz = args.gradient_accumulation_steps * args.batch_size
    if args.test:
        mt = args.test
    output_dir = (
        root_dir
        / f"models/{mt}/{model_name}_lr={args.learning_rate}_bsz={eff_bsz}_epochs={args.num_epochs}_seed={args.seed}/"
    )
    if args.eval:
        output_dir = Path(output_dir.parent) / (output_dir.name + "_eval")
    elif args.test:
        output_dir = Path(output_dir.parent) / (output_dir.name + "_test")
    args.output_dir = output_dir

    # Load the data.
    idx_to_docid = None
    test_or_dev = "test" if args.test else "dev"
    if args.load_sentence_pairs:  # load both source and translations (bilingual)
        train_data = load_corpus_sentence_pairs(args, "train")
        eval_data = load_corpus_sentence_pairs(args, test_or_dev)
    else:  # load only translations (monolingual)
        train_data, _ = load_corpus(args, "train")
        eval_data, idx_to_docid = load_corpus(
            args, test_or_dev, split_docs_by_sentence=args.use_majority_classification
        )

    # Load the model.
    if args.load_model is not None:  # start from a trained model
        print(f"Loading model at {args.load_model}")
        if args.load_sentence_pairs:
            model = XLMRobertaForSequenceClassification.from_pretrained(
                args.load_model, local_files_only=True
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.load_model, local_files_only=True
            )
    else:
        model_name = args.arch
        print(f"Loading LM: {model_name}")
        config = AutoConfig.from_pretrained(
            model_name, num_labels=2, classifier_dropout=args.dropout
        )
        if args.load_sentence_pairs:
            model = BilingualSentenceClassifier(
                XLMRobertaModel.from_pretrained(
                    model_name,
                    config=config,
                    local_files_only=False,
                    add_pooling_layer=False,
                ),
                config.hidden_size,
                dropout=args.dropout,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=config, local_files_only=False
            )

    # Setup Huggingface training arguments.
    training_args = get_training_arguments(args)

    # For logging purposes.
    print("Generated by command:\npython", " ".join(sys.argv))
    print("Logging training settings\n", training_args)

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    ]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=partial(compute_metrics, idx_to_docid=idx_to_docid),
        callbacks=callbacks,
    )

    # Start training/evaluation.
    if args.test or args.eval or args.use_majority_classification:
        mets = trainer.evaluate()
    else:
        mets = trainer.train()
    print("\nInfo:\n", mets, "\n")


if __name__ == "__main__":
    main()