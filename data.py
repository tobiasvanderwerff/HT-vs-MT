import re
import itertools
from pathlib import Path

from transformers import AutoTokenizer

from util import HFDataset


def load_corpus(args, phase, split_docs_by_sentence=False):
    """
    Load sentence-label pairs from disk.

    Args:
        args: arguments as processed by parse_args()
        phase: phase for which data should be loaded
        split_docs_by_sentence: whether to split documents into sentences for the
            purpose of majority classification
    Returns:
        HFDataset returning sentence-label pairs
    """
    if phase not in ("train", "dev", "test"):
        raise ValueError("Phase should be one of 'train', 'dev', 'test'")

    print(f"=> Loading {phase} corpus...")

    corpus_data = []
    root_dir = Path(args.root_dir).resolve()
    mt = mt_name = "google" if args.use_google_data else "deepl"
    if phase == "test":
        mt_name = args.test
    if mt_name.startswith("wmt"):
        mt = "wmt_submissions"
    apdx = "normalized" if args.use_normalized_data else ""
    paths = {
        0: list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.txt")),
        1: (
            list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.deepl.en"))
            + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.en.google"))
        ),
    }  # all the text files per class
    if mt_name == "wmt1":
        paths[1] = [
            root_dir
            / f"data/wmt_submissions/{phase}/{apdx}/newstest2019.Facebook_FAIR.6750.wmt"
        ]
    if mt_name == "wmt2":
        paths[1] = [
            root_dir
            / f"data/wmt_submissions/{phase}/{apdx}/newstest2019.RWTH_Aachen_System.6818.wmt"
        ]
    if mt_name == "wmt3":
        paths[1] = [
            root_dir
            / f"data/wmt_submissions/{phase}/{apdx}/newstest2019.online-X.0.wmt"
        ]
    if mt_name == "wmt4":
        paths[1] = [
            root_dir
            / f"data/wmt_submissions/{phase}/{apdx}/newstest2019.PROMT_NMT_DE-EN.6683.wmt"
        ]

    assert (
        len(paths[0]) != 0 and len(paths[1]) != 0
    ), f"{len(paths[0])}, {len(paths[1])}"

    idx_to_docid = dict() if split_docs_by_sentence else None
    doc_id = 0
    for label, path_lst in paths.items():
        for path in path_lst:
            with open(path, encoding="utf-8") as corpus:
                for line in corpus:
                    if split_docs_by_sentence:
                        # In this case, a single line contains a full document.
                        for seg in line.split(". "):
                            corpus_data.append([f"{seg.rstrip()}.", label])
                            idx_to_docid[len(corpus_data) - 1] = doc_id
                    else:
                        corpus_data.append([line.rstrip(), label])
                    doc_id += 1
    sents, labels = zip(*corpus_data)
    sents = list(sents)

    # Encode the sentences using the HuggingFace tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        args.arch, model_max_length=args.max_length
    )
    sents_enc = tokenizer(sents, padding=True, truncation=True)
    return HFDataset(sents_enc, labels), idx_to_docid


def load_corpus_sentence_pairs(args, phase):
    """
    Loads data from disk, where instead of individual sentences, bilingual sentence
    pairs are loaded (German-English).

    Args:
        args: arguments as processed by parse_args()
        phase: phase for which data should be loaded
    Returns:
        HFDataset returning sentence pair/label pairs
    """
    if phase not in ("train", "dev", "test"):
        raise ValueError("Phase should be one of 'train', 'dev', 'test'")

    print("=> Loading {} corpus...".format(phase))

    _mt_suffixes = [".deepl.en", ".en.google", ".wmt"]

    corpus_data = []
    root_dir = Path(args.root_dir).resolve()
    mt = "google" if args.use_google_data else "deepl"
    if phase == "test":
        mt = args.test
    paths = {
        0: list((root_dir / f"data/{mt}/{phase}/").glob("trans_en*.txt")),
        1: list(
            itertools.chain.from_iterable(
                (root_dir / f"data/{mt}/{phase}/").glob(f"*{sfx}")
                for sfx in _mt_suffixes
            )
        ),
    }
    if mt == "wmt_submissions":
        raise NotImplementedError()

    assert len(paths[0]) != 0 and len(paths[1]) != 0

    # Match source files with files containing translations.
    for label, path_lst in paths.items():
        for path_B in path_lst:
            wmt_year = re.search(r"[0-9]{2}", path_B.name).group(0)
            if path_B.name in [
                f"trans_en_wmt{wmt_year}.txt",
                f"org_de_wmt{wmt_year}.deepl.en",
                f"org_de_wmt{wmt_year}.en.google",
                f"org_de_wmt{wmt_year}.wmt",
            ]:
                # Translation from original text.
                path_A = root_dir / f"data/{mt}/{phase}/org_de_wmt{wmt_year}.txt"
            elif path_B.name in [
                f"trans_de_wmt{wmt_year}.deepl.en",
                f"trans_de_wmt{wmt_year}.en.google",
                f"trans_de_wmt{wmt_year}.wmt",
            ]:
                # Translation from Translationese.
                path_A = root_dir / f"data/{mt}/{phase}/trans_de_wmt{wmt_year}.txt"
            else:  # fail
                raise RuntimeError(
                    f"Unrecognized file name: {path_B.name}. Take a look "
                    f"at the file naming convention in "
                    f"load_corpus_sentence_pairs()` to see why this "
                    f"is unrecognized."
                )
            assert path_A.is_file(), (
                f"Sentence pairs incomplete, missing: {path_A.name}. Make "
                f"sure all translated sentences are coupled with "
                f"a corresponding untranslated sentence."
            )
            with open(path_A, encoding="utf-8") as sents_A:
                with open(path_B, encoding="utf-8") as sents_B:
                    for line_A, line_B in zip(sents_A, sents_B):
                        corpus_data.append([line_A.rstrip(), line_B.rstrip(), label])

    # Encode the sentences using the HuggingFace tokenizer.
    sentsA, sentsB, labels = zip(*corpus_data)
    sentsA, sentsB = list(sentsA), list(sentsB)
    tokenizer = AutoTokenizer.from_pretrained(
        args.arch, model_max_length=args.max_length
    )
    sents_enc = tokenizer(sentsA, sentsB, padding=True, truncation=True)
    return HFDataset(sents_enc, labels)
