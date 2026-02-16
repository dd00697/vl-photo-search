from __future__ import annotations

import numpy as np
import pytest

from vl_photo_search.indexing.embed_clip import _to_numpy_embeddings


class DummyTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def __getitem__(self, item):
        return DummyTensor(self.arr[item])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class DummyPoolerOutput:
    def __init__(self, arr):
        self.pooler_output = DummyTensor(arr)


class DummyHiddenStateOutput:
    def __init__(self, arr):
        self.last_hidden_state = DummyTensor(arr)


def test_to_numpy_embeddings_from_tensor_like() -> None:
    emb = _to_numpy_embeddings(DummyTensor([[1.0, 2.0], [3.0, 4.0]]))
    assert emb.shape == (2, 2)
    assert emb.dtype == np.float32


def test_to_numpy_embeddings_from_pooler_output() -> None:
    emb = _to_numpy_embeddings(DummyPoolerOutput([[0.1, 0.2, 0.3]]))
    assert emb.shape == (1, 3)
    assert emb.dtype == np.float32


def test_to_numpy_embeddings_from_last_hidden_state_cls_token() -> None:
    arr = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    emb = _to_numpy_embeddings(DummyHiddenStateOutput(arr))
    assert emb.shape == (2, 3)
    np.testing.assert_array_equal(emb, arr[:, 0, :].astype(np.float32))


def test_to_numpy_embeddings_invalid_output_type() -> None:
    with pytest.raises(TypeError):
        _to_numpy_embeddings(object())
