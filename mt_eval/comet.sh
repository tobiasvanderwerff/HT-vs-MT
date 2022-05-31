#!/bin/bash
#SBATCH --time=00:14:59
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50G

# Evaluate COMET on WMT08-19.

set -euxo pipefail

lang=en
mt=google-bad-api

# DeepL
if [[ $mt = "deepl" ]]; then
    out_dir=/data/pg-macocu/MT_vs_HT/wmt_data/deepL/deepl_20211101/  # Files produced by model
    ref_dir=/data/pg-macocu/MT_vs_HT/wmt_data/deepL/WMT08-20_for_deepl/  # Gold standard files
    src_dir=/data/pg-macocu/MT_vs_HT/wmt_data/deepL/WMT08-20_for_deepl/ # Source files for translation
elif [[ $mt = "google" ]]; then
    out_dir=/data/pg-macocu/MT_vs_HT/wmt_data/google/  # Files produced by model
    ref_dir=/data/pg-macocu/MT_vs_HT/wmt_data/deepL/WMT08-20_for_deepl/  # Gold standard files
    src_dir=/data/pg-macocu/MT_vs_HT/wmt_data/deepL/WMT08-20_for_deepl/ # Source files for translation
elif [[ $mt = "google-bad-api" ]]; then
    out_dir=/data/pg-macocu/MT_vs_HT/wmt_data/google_bad_api/  # Files produced by model
    ref_dir=/data/pg-macocu/MT_vs_HT/wmt_data/deepL/WMT08-20_for_deepl/  # Gold standard files
    src_dir=/data/pg-macocu/MT_vs_HT/wmt_data/deepL/WMT08-20_for_deepl/ # Source files for translation
else
    echo "$mt is not a valid mt."
    exit 1
fi

out=$TMPDIR/out.txt
ref=$TMPDIR/ref.txt
src=$TMPDIR/src.txt

# Concatentate all files within each category (org, ref, src).
cat $out_dir/org_de_wmt{08..19}* >> $out
cat $ref_dir/trans_en_wmt{08..19}.txt >> $ref
cat $src_dir/org_de_wmt{08..19}.txt >> $src

comet-score -s $src -t $out -r $ref > ${mt}.eval.comet
