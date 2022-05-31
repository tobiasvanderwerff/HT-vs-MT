"""Count the mean number of subword tokens over all data instances."""

import argparse
import statistics
from pathlib import Path

from transformers import AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Location of the "
        "data directory. Token counts are performed for each file in the directory.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="microsoft/deberta-v3-large",
        help=(
            "Architecture to use, e.g. `bert-base-cased`. This "
            "determines what tokenizer is used."
        ),
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length, which is used to calculate the "
        "percentage of tokens that would be discarded if truncation is "
        "applied.",
    )
    args = parser.parse_args()

    print("Mean number of subword tokens per file:")

    all_token_counts = []
    max_count, min_count = float("-inf"), float("inf")
    tokenizer = AutoTokenizer.from_pretrained(args.arch, model_max_length=None)
    for doc_pth in sorted(args.data_dir.iterdir()):
        token_count, discard_prcnt = [], []
        with open(doc_pth, "r", encoding="utf-8") as corpus:
            for line in corpus:
                line = line.strip()
                tokenized = tokenizer(line, padding=False, truncation=False)
                cnt = len(tokenized["input_ids"])
                if cnt > max_count:
                    max_count = cnt
                if cnt < min_count:
                    min_count = cnt
                discard_prcnt.append(100 * max(0, cnt - args.max_seq_len) / cnt)
                token_count.append(cnt)
                all_token_counts.append(cnt)
            mean_discarded_prcnt = int(statistics.mean(discard_prcnt))
            mean_ntokens = int(statistics.mean(token_count))
            print(
                f"{doc_pth.name}:\t{mean_ntokens} (avg {mean_discarded_prcnt}% discarded)"
            )

    global_mean_ntokens = int(statistics.mean(all_token_counts))
    global_median_ntokens = int(statistics.median(all_token_counts))

    print(f"\nOverall mean: {global_mean_ntokens} tokens")
    print(f"Overall median: {global_median_ntokens} tokens")
    print(f"Min: {min_count}")
    print(f"Max: {max_count}")
