"""Unit tests for pipeline components."""
import torch
import pytest
from src.models.block_encoder import TransformerBlockEncoder, MeanBlockEncoder
from src.models.graph_encoder import GATGraphEncoder
from src.models.external_encoder import ExternalCallEncoder
from src.models.gated_fusion import GatedFusion
from src.models.decoder import GRUDecoder
from src.evaluation.metrics import compute_subtoken_f1, split_name


class TestBlockEncoder:
    def test_transformer_output_shape(self):
        enc = TransformerBlockEncoder(vocab_size=100, embed_dim=32,
                                       num_layers=1, num_heads=2, output_dim=64)
        tokens = torch.randint(0, 100, (2, 5, 10))  # batch=2, 5 blocks, 10 tokens
        out = enc(tokens)
        assert out.shape == (2, 5, 64)

    def test_mean_output_shape(self):
        enc = MeanBlockEncoder(vocab_size=100, embed_dim=32, output_dim=64)
        tokens = torch.randint(0, 100, (2, 5, 10))
        out = enc(tokens)
        assert out.shape == (2, 5, 64)


class TestGraphEncoder:
    def test_output_shape(self):
        enc = GATGraphEncoder(input_dim=64, hidden_dim=64, output_dim=128,
                               num_layers=2, num_heads=2)
        x = torch.randn(10, 64)  # 10 nodes
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]], dtype=torch.long)
        batch = torch.tensor([0]*5 + [1]*5)
        f, block_embs, weights = enc(x, edge_index, batch)
        assert f.shape == (2, 128)  # 2 graphs


class TestExternalEncoder:
    def test_with_calls(self):
        enc = ExternalCallEncoder(vocab_size=50, embed_dim=32,
                                   hidden_dim=16, output_dim=64)
        ext_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])  # batch=2
        c = enc(ext_ids)
        assert c.shape == (2, 64)

    def test_no_calls(self):
        enc = ExternalCallEncoder(vocab_size=50, embed_dim=32,
                                   hidden_dim=16, output_dim=64)
        ext_ids = torch.tensor([[0, 0, 0]])  # No calls
        c = enc(ext_ids)
        assert c.shape == (1, 64)


class TestGatedFusion:
    def test_gate_shape(self):
        fusion = GatedFusion(input_dim=64)
        f = torch.randn(2, 64)
        c = torch.randn(2, 64)
        z, g = fusion(f, c)
        assert z.shape == (2, 64)
        assert g.shape == (2, 64)
        assert (g >= 0).all() and (g <= 1).all()


class TestDecoder:
    def test_training_output(self):
        dec = GRUDecoder(vocab_size=100, embed_dim=32, hidden_dim=64)
        z = torch.randn(2, 64)
        decoder_input = torch.randint(1, 100, (2, 5))
        logits = dec(z, decoder_input)
        assert logits.shape == (2, 5, 100)

    def test_beam_search(self):
        dec = GRUDecoder(vocab_size=100, embed_dim=32, hidden_dim=64, max_length=10)
        z = torch.randn(1, 64)
        tokens, score = dec.generate(z, sos_id=1, eos_id=2, beam_width=3)
        assert isinstance(tokens, list)


class TestMetrics:
    def test_perfect_match(self):
        assert compute_subtoken_f1("socket_init", "socket_init") == 1.0

    def test_partial_match(self):
        f1 = compute_subtoken_f1("read_config", "parse_config")
        assert 0 < f1 < 1  # "config" matches, "read" vs "parse" doesn't

    def test_no_match(self):
        assert compute_subtoken_f1("calc_hash", "compute_checksum") == 0.0

    def test_split_name(self):
        assert split_name("socket_init") == ["socket", "init"]
        assert split_name("handleClient") == ["handle", "client"]
        assert split_name("getHTTPResponse") == ["get", "http", "response"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
