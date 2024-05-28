import pytest

from sentence_transformers import SentenceTransformer


@pytest.mark.parametrize(
    ("revision", "expected_base_revision"),
    [
        ("f3cb857cba53019a20df283396bcca179cf051a4", "f3cb857cba53019a20df283396bcca179cf051a4"),
        ("f3cb857", "f3cb857"),
        ("main", "valid-revision"),
        (None, "valid-revision"),
    ],
)
def test_model_card_data(revision, expected_base_revision) -> None:
    model_name = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = SentenceTransformer(model_name, revision=revision)

    assert model.model_card_data.base_model == model_name
    if expected_base_revision == "valid-revision":
        assert model.model_card_data.base_model_revision
        assert len(model.model_card_data.base_model_revision) == 40
    else:
        assert model.model_card_data.base_model_revision == expected_base_revision
