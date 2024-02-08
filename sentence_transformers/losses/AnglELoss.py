from sentence_transformers import losses, SentenceTransformer, util


class AnglELoss(losses.CoSENTLoss):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0):
        """
        This class implements AnglE (Angle Optimized) loss.
        This is a modification of :class:`CoSENTLoss`, designed to address the following issue:
        The cosine function's gradient approaches 0 as the wave approaches the top or bottom of its form.
        This can hinder the optimization process, so AnglE proposes to instead optimize the angle difference
        in complex space in order to mitigate this effect.

        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(k,l)-s(i,j))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition. This is the same as CoSENTLoss, with a different
        similarity function.

        :param model: SentenceTransformerModel
        :param scale: Output of similarity function is multiplied by scale value. Represents the inverse temperature.

        References:
            - For further details, see: https://arxiv.org/abs/2309.12871v1

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Relations:
            - :class:`CoSENTLoss` is AnglELoss with ``pairwise_cos_sim`` as the metric, rather than ``pairwise_angle_sim``.
            - :class:`CosineSimilarityLoss` seems to produce a weaker training signal than ``CoSENTLoss`` or ``AnglELoss``.

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses
                from sentence_transformers.readers import InputExample

                model = SentenceTransformer('bert-base-uncased')
                train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=1.0),
                        InputExample(texts=['My third sentence', 'Unrelated sentence'], label=0.3)]

                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
                train_loss = losses.AnglELoss(model=model)
        """
        super().__init__(model, scale, similarity_fct=util.pairwise_angle_sim)
