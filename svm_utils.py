import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tabulate import tabulate


def parse_args_svm():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./experiments/8",
        help="Root directory for the MT vs HT experiment.",
    )
    parser.add_argument(
        "-tf",
        "--tfidf",
        action="store_true",
        help="Use the TF-IDF vectorizer instead of CountVectorizer",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=["nb", "svm"],
        default="svm",
        type=str,
        help="What algorithm are we using? Currently only NB or SVM",
    )
    parser.add_argument(
        "-cv",
        "--cross_validate",
        default=5,
        type=int,
        help="How many folds for CV? Only do when no test file is added",
    )
    parser.add_argument(
        "-md",
        "--min_df",
        default=5,
        type=int,
        help="Minimum amount a feature should occur before being added",
    )
    parser.add_argument(
        "-f", "--features", action="store_true", help="Print best features per class"
    )
    parser.add_argument(
        "-owf",
        "--only_word_features",
        action="store_true",
        help="If added, we use only the word features as defined in a dict",
    )
    parser.add_argument(
        "-cm",
        "--confusion",
        default="",
        type=str,
        help="Save plot of confusion matrix here, if not added do not plot",
    )
    parser.add_argument(
        "-ovr",
        "--one_vs_rest",
        action="store_true",
        help="Do one vs rest classification instead of one vs one (default)",
    )
    parser.add_argument(
        "-pr",
        "--probabilities",
        action="store_true",
        help="Print the probabilities to a file instead of the labels for -tnl",
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
        "a Moses normalization script to them. Right now only works "
        "for monolingual sentences",
    )

    args = parser.parse_args()
    if args.features and args.algorithm != "svm":
        raise ValueError("Function --features is only implemented for -a svm")
    return args


def feature_count(vectorizer, X_train):
    """For each feature get its name, the number of docs it appears in and the total
    amount."""
    count_dict = {}
    # Get the feature matrix
    matrix = vectorizer.fit_transform(X_train)
    # Loop over names and full count
    for name, count in zip(
        vectorizer.get_feature_names(), matrix.sum(axis=0).tolist()[0]
    ):
        count_dict[name] = count
    return count_dict


def print_division(label_names, labels):
    """Print label division of training set."""
    print("\nLabel division:")
    print(tabulate([[label, labels.count(label)] for label in label_names]))
    print()


def print_best_features(vectorizer, clf, X_train, only_words):
    """Prints features with the highest coefficient values, per class."""
    # Check if we only want to print features that are English words.
    # We also want to get the number of docs the feature occurs in (and total amount).
    count_dict = feature_count(vectorizer, X_train)
    # Now get the best features and print them.
    num_features = 8
    labels = clf.named_steps["cls"].classes_
    feature_names = vectorizer.get_feature_names_out()
    for i, class_label in enumerate(labels):
        top = np.argsort(clf.named_steps["cls"].coef_[i])
        # Get the best features, order from best to worst.
        # Select a bit more because we might filter non-English words later.
        sort_top = top[-(num_features) * 10 :][::-1]
        # Print features most indicative of this class.
        print("\nBest features for " + class_label + ":\n")
        done = []
        for j in sort_top:
            # Stop if we output enough features already.
            if len(done) >= num_features:
                break
            print(
                feature_names[j],
                round(clf.named_steps["cls"].coef_[i][j], 2),
                "({0})".format(round(count_dict[feature_names[j]], 1)),
            )
            done.append(feature_names[j])
        # Command to show words as just a list.
        print("\n" + class_label + ":", ", ".join(done) + "\n")
        if i == 0:
            break  # for binary classification


def load_data(root_dir, phase, use_google_data=False, use_normalized_data=False):
    """Loads a HT vs. MT dataset."""
    if phase not in ("train", "dev", "test"):
        raise ValueError("Phase should be one of 'train', 'dev', 'test'")

    print("=> Loading {} corpus...".format(phase))

    corpus_data = []
    root_dir = Path(root_dir).resolve()
    mt = "google" if use_google_data else "deepl"
    apdx = "normalized" if use_normalized_data else ""
    print(f"MT: {mt}")
    paths = {
        1: list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.txt")),
        0: list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.deepl.en"))
        + list((root_dir / f"data/{mt}/{phase}/{apdx}").glob("*.en.google")),
    }

    assert (
        len(paths[0]) != 0 and len(paths[1]) != 0
    ), f"{len(paths[0])}, {len(paths[1])}"
    for label, path_lst in paths.items():
        for path in path_lst:
            with open(path, encoding="utf-8") as corpus:
                for line in corpus:
                    corpus_data.append([line.rstrip(), str(label)])
    sents, labels = zip(*corpus_data)
    sents, labels = list(sents), list(labels)
    return sents, labels


# Taken directly from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def plot_confusion_matrix(
    cm, target_names, save_to, title="Confusion matrix", cmap=None, normalize=True
):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    # accuracy = np.trace(cm) / np.sum(cm).astype('float')
    plt.rcParams.update({"font.size": 12.5})
    if cmap is None:
        cmap = plt.get_cmap("Purples")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap, vmax=300)
    # plt.title(title)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    thresh = 275
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label", size=16)
    plt.xlabel("Predicted label", size=16)
    plt.savefig(save_to, bbox_inches="tight")
