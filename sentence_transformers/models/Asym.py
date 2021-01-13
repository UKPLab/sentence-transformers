from torch import Tensor
from torch import nn
from typing import List, Dict
import os
import json
from ..util import import_from_string
from collections import OrderedDict
from typing import List, Dict, Optional, Union, Tuple

class Asym(nn.Sequential):
    def __init__(self, sub_modules: Dict[str, List[nn.Module]], allow_empty_key: bool = True):
        """
        This model allows to create asymmetric SentenceTransformer models, that apply different models depending on the specified input key.

        In the below example, we create two different Dense models for 'query' and 'doc'. Text that is passed as {'query': 'My query'} will
        be passed along along the first Dense model, and text that will be passed as {'doc': 'My document'} will use the other Dense model.

        Note, that when you call encode(), that only inputs of the same type can be encoded. Mixed-Types cannot be encoded.

        Example::
            word_embedding_model = models.Transformer(model_name)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            asym_model = models.Asym({'query': [models.Dense(word_embedding_model.get_word_embedding_dimension(), 128)], 'doc': [models.Dense(word_embedding_model.get_word_embedding_dimension(), 128)]})
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, asym_model])

            model.encode([{'query': 'Q1'}, {'query': 'Q2'}]
            model.encode([{'doc': 'Doc1'}, {'doc': 'Doc2'}]

            #You can train it with InputExample like this. Note, that the order must always be the same:
            train_example = InputExample(texts=[{'query': 'Train query', 'doc': 'Doc query'}], label=1)


        :param sub_modules: Dict in the format str -> List[models]. The models in the specified list will be applied for input marked with the respective key.
        :param allow_empty_key: If true, inputs without a key can be processed. If false, an exception will be thrown if no key is specified.
        """
        self.sub_modules = sub_modules
        self.allow_empty_key = allow_empty_key

        ordered_dict = OrderedDict()
        for name, models in sub_modules.items():
            if not isinstance(models, List):
                models = [models]

            for idx, model in enumerate(models):
                ordered_dict[name+"-"+str(idx)] = model
        super(Asym, self).__init__(ordered_dict)


    def forward(self, features: Dict[str, Tensor]):
        if 'text_keys' in features and len(features['text_keys']) > 0:
            text_key = features['text_keys'][0]
            for model in self.sub_modules[text_key]:
                features = model(features)
        elif not self.allow_empty_key:
            raise ValueError('Input did not specify any keys and allow_empty_key is False')

        return features

    def get_sentence_embedding_dimension(self) -> int:
        raise NotImplementedError()

    def save(self, output_path):
        model_lookup = {}
        model_types = {}
        model_structure = {}

        for name, models in self.sub_modules.items():
            model_structure[name] = []
            for model in models:
                model_id = str(id(model))+'_'+type(model).__name__
                model_lookup[model_id] = model
                model_types[model_id] = type(model).__module__
                model_structure[name].append(model_id)

        for model_id, model in model_lookup.items():
            model_path = os.path.join(output_path, str(model_id))
            os.makedirs(model_path, exist_ok=True)
            model.save(model_path)

        with open(os.path.join(output_path, 'config.json'), 'w', encoding='utf8') as fOut:
            json.dump({'types': model_types, 'structure': model_structure,
                       'parameters': {'allow_empty_key': self.allow_empty_key}},
                      fOut, indent=2)

    def tokenize(self, texts: Union[List[str], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        if not isinstance(texts[0], dict):
            raise AttributeError("Asym. model requires that texts are passed as dicts: {'key': 'text'}")


        module_key = None

        for lookup in texts:
            text_key, text = next(iter(lookup.items()))
            if module_key is None:
                module_key = text_key

            assert text_key == module_key   #Mixed batches are not allowed
        return self.sub_modules[module_key][0].tokenize(texts)


    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        modules = {}
        for model_id, model_type in config['types'].items():
            module_class = import_from_string(model_type)
            module = module_class.load(os.path.join(input_path, model_id))
            modules[model_id] = module

        model_structure = {}
        for key_name, models_list in config['structure'].items():
            model_structure[key_name] = []
            for model_id in models_list:
                model_structure[key_name].append(modules[model_id])

        model = Asym(model_structure, **config['parameters'])
        return model