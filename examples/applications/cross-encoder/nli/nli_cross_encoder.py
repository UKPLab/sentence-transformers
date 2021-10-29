from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
model_path="trained_model_name or path"
model=CrossEncoder(model_path)
preds=model.predict(["the city is beautiful","the city is ugly"])
labels=["contradiction","entailment","neutral"]
print(labels[np.argmax(preds)])