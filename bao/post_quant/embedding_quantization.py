import numpy as np 
from sentence_transformers.quantization import quantize_embeddings 

def post_quantize_embeddings(embeddings:np.ndarray,precision:str, **kwargs)->np.ndarray:
    if precision == "binary":
        return quantize_embeddings(embeddings, precision = "binary")
    else:
        raise ValueError(f"{precision} quantization not yet supported!")