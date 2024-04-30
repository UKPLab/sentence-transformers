source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
PYTHONPATH=. python james/msmarco/score.py \
  --residing_folder data/results/post_quant \
  --knn_index_file knn.npy \
  --output_file inference.tsv
