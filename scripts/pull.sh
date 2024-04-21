mkdir -p data
cd data
if [ ! \( -d queries \) ]; then
  echo "Pulling Queries"
  mkdir -p queries
  if [ ! \( -f queries.tar.gz \) ]; then
    wget https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
  fi
  tar -xvzf queries.tar.gz -C queries &&
  rm queries.tar.gz
fi
if [ ! \( -d collection \) ]; then
  echo "Pulling Collection"
  mkdir -p collection
  if [ ! \( -f collection.tar.gz \) ]; then
    wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
  fi
  tar -xvzf collection.tar.gz -C collection &&
  rm collection.tar.gz
fi
if [ ! \( -d top1000 \) ]; then
  echo "Pulling Collection"
  mkdir -p top1000
  if [ ! \( -f top1000.dev.tar.gz \) ]; then
    wget https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz
  fi
  tar -xvzf top1000.dev.tar.gz -C top1000 &&
  rm top1000.dev.tar.gz &&
  mv data/top1000/top1000.dev data/top1000/top1000.dev.tsv
fi
if [ ! \( -d qrels \) ]; then
  echo "Pulling Qrels"
  mkdir -p qrels
  if [ ! \( -f qrels.dev.tsv \) ]; then
    wget https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv
  fi
fi
# wget https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
# wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
# wget https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz