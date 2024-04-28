source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
PYTHONPATH=. python james/embed/embed.py \
  --output_file data/results/baseline/trels_embeddings.npy \
  --batch_size 256 \
  --tsv data/qrels/qrels_with_text.tsv

