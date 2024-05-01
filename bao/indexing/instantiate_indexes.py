from bao.indexing.custom_indexes import USEARCHIndex, FAISSIndexFlat, FAISSIndexSHNSW, FAISSIndexIVF
from bao.utils import is_binary, is_float32, is_scalar

PATH_INDEX_FLAT_FLOAT = "data/indexes/flat_float.index"
PATH_INDEX_FLAT_BINARY = "data/indexes/flat_binary.index"
PATH_INDEX_FLAT_SCALAR = "data/indexes/flat_scalar.index"
PATH_INDEX_IVF_FLOAT = "data/indexes/ivf_float.index"
PATH_INDEX_IVF_BINARY = "data/indexes/ivf_binary.index"
PATH_INDEX_IVF_SCALAR = "data/indexes/ivf_scalar.index"
PATH_INDEX_HNSW_SCALAR = "data/indexes/hnsw_scalar.index"
PATH_USEARCH_SCALAR = "data/indexes/usearch_scalar.index"


def get_index(dim:int, precision:str, use_usearch:bool, use_flat:bool, use_ivf:bool):
    
    if use_usearch:
        return USEARCHIndex(dim, PATH_USEARCH_SCALAR, exact=True)
    
    if use_flat: 
        if is_float32(precision):
            default_path = PATH_INDEX_IVF_FLOAT # should NEVER be used
        elif is_binary(precision):
            default_path = PATH_INDEX_FLAT_BINARY 
        elif is_scalar(precision):
            default_path = PATH_INDEX_FLAT_SCALAR 
        else:
            raise ValueError(f"precision {precision} not supported")
        
        return FAISSIndexFlat(dim, default_path, precision)
    
    if not use_flat:
        if is_binary(precision):
            return FAISSIndexIVF(dim, PATH_INDEX_IVF_BINARY, precision)
        elif is_float32(precision):
            return FAISSIndexIVF(dim,PATH_INDEX_IVF_FLOAT, precision)
        elif is_scalar(precision):
            if use_ivf:
                return FAISSIndexIVF(dim, PATH_INDEX_IVF_SCALAR, precision) 
            else:
                return FAISSIndexSHNSW(dim, PATH_INDEX_HNSW_SCALAR, precision)
        else:
            raise ValueError(f"precision {precision} not supported")
    
    else:
        return ValueError("Something's wrong in your inputs")



