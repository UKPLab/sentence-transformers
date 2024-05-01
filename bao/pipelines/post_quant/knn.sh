source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
PYTHONPATH=. python bao/retrieve/knn.py \
  --residing_folder data/results/baseline \
  --collection_file collection_embeddings.npy \
  --query_file trels_embeddings.npy \
  --output_folder data/results/quant_binary_flat \
  --output_file knn.npy \
  --output_file_scores knn_D.npy \
  --post_quant binary\
  --use_flat \
  # --nlist 10000 \
  # --nprobe 200 \
  

PYTHONPATH=. python james/msmarco/score.py \
  --residing_folder data/results/quant_binary_flat \
  --knn_index_file knn.npy \
  --output_file inference.tsv