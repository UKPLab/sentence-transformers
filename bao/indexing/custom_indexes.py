from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import faiss
from usearch.index import Index
from bao.utils import measure_time, is_binary


class IndexBase(ABC):
    def __init__(self, index_dim:int, default_path: Optional[str] = None) -> None:
        self.index = None
        self.index_dim = index_dim 
        self.default_path = default_path
        self.instanciate_index(default_path)

    @abstractmethod
    @measure_time
    def train_and_add(self, corpus_embeddings):
        pass
    
    @abstractmethod 
    @measure_time
    def search(self, queries, k):
        pass

    @abstractmethod
    def save_index(self, index_path):
        pass 

    @abstractmethod
    def instanciate_index(self, index_path):
        pass

class FAISSIndex(IndexBase):
    def __init__(self, index_dim: int, default_path: Optional[str] = None, corpus_precision:str = "float32") -> None:
        self.is_binary = is_binary(corpus_precision)
        super().__init__(index_dim, default_path)

    @measure_time
    def train_and_add(self, corpus_embeddings):
        assert self.index is not None
        self.index.add(corpus_embeddings) 
    
    @measure_time
    def search(self, queries, k):
        assert self.index is not None 
        return self.index.search(queries, k)

    def save_index(self, index_path):
        assert self.index is not None
        if self.is_binary:
            faiss.write_index_binary(self.index, index_path)
        else:
            faiss.write_index(self.index, index_path) 

    def instanciate_index(self, index_path):
        try:
            self.index = faiss.read_index_binary(index_path) if self.is_binary else faiss.read_index(index_path)
        except:
            self.index = None
    

class FAISSIndexFlat(FAISSIndex):
    def __init__(self, index_dim: int, default_path:Optional[str] = None, corpus_precision:str = "float32") -> None:
        super().__init__(index_dim, default_path, corpus_precision)
        if self.index is None:
            self.index =  faiss.IndexBinaryFlat(index_dim * 8) if self.is_binary else faiss.IndexFlatIP(index_dim) 


class FAISSIndexIVF(FAISSIndex):
    def __init__(self, index_dim: int, default_path: Optional[str] = None, corpus_precision:str = "float32") -> None:
        super().__init__(index_dim, default_path, corpus_precision)
        if self.index is None:
            n_list = 10000 # hard-coded for now
            n_probe = 200
            quantizer = faiss.IndexBinaryFlat(index_dim * 8) if self.is_binary else faiss.IndexFlatIP(index_dim)
            self.index =  faiss.IndexBinaryIVF(quantizer, index_dim * 8, n_list) if self.is_binary else faiss.IndexIVFFlat(quantizer, index_dim, n_list)
            self.index.nprobe = n_probe

    @measure_time
    def train_and_add(self, corpus_embeddings):
        assert self.index is not None
        self.index.train(corpus_embeddings)
        self.index.add(corpus_embeddings)
  
class FAISSIndexSHNSW(FAISSIndex):
    def __init__(self, index_dim: int, default_path: Optional[str] = None, corpus_precision:str = "uint8") -> None:
        super().__init__(index_dim, default_path, corpus_precision)
        assert not self.is_binary
        if self.index is None:
            self.index = faiss.IndexHNSWFlat(index_dim * 8, 16) # 16 is hardcoded for now
    

class USEARCHIndex(IndexBase):
    def __init__(self, index_dim: int, default_path: Optional[str] = None, exact:bool = True) -> None:
        corpus_precision = "i8" # only one needed and supported for now
        self.exact = exact
        super().__init__(index_dim, default_path)
        if self.index is None:
            self.index = Index(
                ndim=index_dim,
                metric="ip",
                dtype=corpus_precision,
            )
    @measure_time
    def train_and_add(self, corpus_embeddings):
        assert self.index is not None
        self.index.add(np.arange(len(corpus_embeddings)), corpus_embeddings)
    @measure_time
    def search(self, queries, k):
        assert self.index is not None
        matches = self.index.search(queries, count=k, exact=self.exact)
        scores = matches.distances
        indices = matches.keys
        return scores, indices

    def save_index(self, index_path):
        assert self.index is not None
        self.index.save(index_path)
    
    def instanciate_index(self, index_path):
        try:
            self.index = Index.restore(index_path, view = True) #hopefully enough for 1step
        except:
            self.index = None






