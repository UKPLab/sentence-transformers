source ~/miniconda3/etc/profile.d/conda.sh && 
conda activate $ENV_NAME &&
cd $SENTENCE_TRANSFORMERS && 
PYTHONPATH=. python james/retrieve/knn.py \
  --residing_folder data/results/baseline \
  --collection_file collection_embeddings.npy \
  --query_file trels_embeddings.npy \
  --output_file data/results/truncation/knn_96.npy \
  --output_file_scores data/results/baseline/knn_96_D.npy \
  --nlist 10000 \
  --nprobe 200 \
  --d 384 \
  --truncation_d 96

