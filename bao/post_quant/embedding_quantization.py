import numpy as np 
from sentence_transformers.quantization import quantize_embeddings 

def post_quantize_embeddings(embeddings:np.ndarray,precision:str, **kwargs)->np.ndarray:
    if precision.endswith("float32"):
        return embeddings
    elif precision.endswith("binary"):
        return quantize_embeddings(embeddings, precision = "ubinary")
    elif precision.endswith("int8") or precision.endswith("scalar"):
        return quantize_embeddings(embeddings, calibration_embeddings=embeddings, precision="uint8") #TODO: to customize, so far we take all embeddings for calibration
    else:
        raise ValueError(f"{precision} quantization not yet supported!")