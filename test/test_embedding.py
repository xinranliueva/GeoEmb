import torch
import tempfile


def test_embedding_save_load():

    emb = torch.randn(100, 16)

    with tempfile.NamedTemporaryFile() as tmp:

        torch.save(emb, tmp.name)

        loaded = torch.load(tmp.name, weights_only=True)

        assert emb.shape == loaded.shape