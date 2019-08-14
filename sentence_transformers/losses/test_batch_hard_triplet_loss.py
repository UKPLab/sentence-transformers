import numpy as np
import torch
from sentence_transformers.losses import BatchHardTripletLoss

# Test-suite from https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/tests/test_triplet_loss.py
# Skipped the `test_gradients_pairwise_distances()` test since it's trivial to see if your model loss turns NaN
# and porting it proved more difficult than expected.

def pairwise_distance_np(feature, squared=False):
    """Computes the pairwise distance matrix in numpy.
    Args:
        feature: 2-D numpy array of size [number of data, feature dimension]
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix; else, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: 2-D numpy array of size
                            [number of data, number of data].
    """
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
    if squared:
        upper_tri_pdists **= 2.
    num_data = feature.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
        pairwise_distances.diagonal())
    return pairwise_distances

def test_pairwise_distances():
    """Test the pairwise distances function."""
    num_data = 64
    feat_dim = 6

    embeddings = np.random.randn(num_data, feat_dim).astype(np.float32)
    embeddings[1] = embeddings[0]  # to get distance 0

    for squared in [True, False]:
        res_np = pairwise_distance_np(embeddings, squared=squared)
        res_pt = BatchHardTripletLoss._pairwise_distances(torch.from_numpy(embeddings), squared=squared)
        assert np.allclose(res_np, res_pt)

def test_pairwise_distances_are_positive():
    """Test that the pairwise distances are always positive.
    Use a tricky case where numerical errors are common.
    """
    num_data = 64
    feat_dim = 6

    # Create embeddings very close to each other in [1.0 - 2e-7, 1.0 + 2e-7]
    # This will encourage errors in the computation
    embeddings = 1.0 + 2e-7 * np.random.randn(num_data, feat_dim).astype(np.float32)
    embeddings[1] = embeddings[0]  # to get distance 0

    for squared in [True, False]:
        res_tf = BatchHardTripletLoss._pairwise_distances(torch.from_numpy(embeddings), squared=squared)
        assert res_tf[res_tf < 0].sum() == 0


def test_triplet_mask():
    """Test function _get_triplet_mask."""
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    mask_np = np.zeros((num_data, num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                distinct = (i != j and i != k and j != k)
                valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                mask_np[i, j, k] = (distinct and valid)

    mask_tf_val = BatchHardTripletLoss._get_triplet_mask(torch.from_numpy(labels))
    assert np.allclose(mask_np, mask_tf_val)

def test_anchor_positive_triplet_mask():
    """Test function _get_anchor_positive_triplet_mask."""
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    mask_np = np.zeros((num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            distinct = (i != j)
            valid = labels[i] == labels[j]
            mask_np[i, j] = (distinct and valid)

    mask_tf_val = BatchHardTripletLoss._get_anchor_positive_triplet_mask(torch.from_numpy(labels))

    assert np.allclose(mask_np, mask_tf_val)

def test_anchor_negative_triplet_mask():
    """Test function _get_anchor_negative_triplet_mask."""
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    mask_np = np.zeros((num_data, num_data))
    for i in range(num_data):
        for k in range(num_data):
            distinct = (i != k)
            valid = (labels[i] != labels[k])
            mask_np[i, k] = (distinct and valid)

    mask_tf_val = BatchHardTripletLoss._get_anchor_negative_triplet_mask(torch.from_numpy(labels))

    assert np.allclose(mask_np, mask_tf_val)

def test_simple_batch_all_triplet_loss():
    """Test the triplet loss with batch all triplet mining in a simple case.
    There is just one class in this super simple edge case, and we want to make sure that
    the loss is 0.
    """
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 1

    embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)
    labels, embeddings = torch.from_numpy(labels), torch.from_numpy(embeddings)

    for squared in [True, False]:
        loss_np = 0.0

        # Compute the loss in TF.
        loss_tf_val, fraction_val = BatchHardTripletLoss.batch_all_triplet_loss(labels, embeddings, margin, squared=squared)

        assert np.allclose(loss_np, loss_tf_val)
        assert np.allclose(fraction_val, 0.0)


def test_batch_all_triplet_loss():
    """Test the triplet loss with batch all triplet mining"""
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 5

    embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    for squared in [True, False]:
        pdist_matrix = pairwise_distance_np(embeddings, squared=squared)

        loss_np = 0.0
        num_positives = 0.0
        num_valid = 0.0
        for i in range(num_data):
            for j in range(num_data):
                for k in range(num_data):
                    distinct = (i != j and i != k and j != k)
                    valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                    if distinct and valid:
                        num_valid += 1.0

                        pos_distance = pdist_matrix[i][j]
                        neg_distance = pdist_matrix[i][k]

                        loss = np.maximum(0.0, pos_distance - neg_distance + margin)
                        loss_np += loss

                        num_positives += (loss > 0)

        loss_np /= num_positives

        # Compute the loss in TF.
        loss_tf_val, fraction_val = BatchHardTripletLoss.batch_all_triplet_loss(torch.from_numpy(labels), torch.from_numpy(embeddings), margin, squared=squared)
        assert np.allclose(loss_np, loss_tf_val)
        assert np.allclose(num_positives / num_valid, fraction_val)

def test_batch_hard_triplet_loss():
    """Test the triplet loss with batch hard triplet mining"""
    num_data = 50
    feat_dim = 6
    margin = 0.2
    num_classes = 5
    min_class = 100

    embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(min_class, min_class+num_classes, size=(num_data)).astype(np.float32)

    for squared in [True, False]:
        pdist_matrix = pairwise_distance_np(embeddings, squared=squared)

        loss_np = 0.0
        for i in range(num_data):
            # Select the hardest positive
            max_pos_dist = np.max(pdist_matrix[i][labels == labels[i]])

            # Select the hardest negative
            min_neg_dist = np.min(pdist_matrix[i][labels != labels[i]])


            loss = np.maximum(0.0, max_pos_dist - min_neg_dist + margin)
            loss_np += loss

        loss_np /= num_data

        # Compute the loss in TF.
        loss_tf_val = BatchHardTripletLoss.batch_hard_triplet_loss(torch.from_numpy(labels), torch.from_numpy(embeddings), margin, squared=squared)
        assert np.allclose(loss_np, loss_tf_val)

if __name__ == '__main__':
    test_pairwise_distances()
    test_pairwise_distances_are_positive()
    test_triplet_mask()
    test_anchor_positive_triplet_mask()
    test_anchor_negative_triplet_mask()
    test_batch_hard_triplet_loss()
    print("--TESTS done ---")