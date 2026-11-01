"""F4 packing on a tiny random Gemma 4: packed readout logits == the stock unpacked forward (HF's own masks)."""
import random

import pytest
import torch
from PIL import Image

from bio_waila.models.jev_packed import block_masks, collate_packed, pack, packed_readout_logits

IMG, BOI, EOI = 60, 61, 62


def tiny_gemma4(bidir=None, window=8, impl="sdpa", kv_shared=2):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    t = dict(vocab_size=64, vocab_size_per_layer_input=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
             num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
             hidden_size_per_layer_input=8, num_kv_shared_layers=kv_shared, sliding_window=window,
             layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
             use_bidirectional_attention=bidir, pad_token_id=0, final_logit_softcapping=30.0)
    v = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
             head_dim=16)
    c = Gemma4Config(text_config=t, vision_config=v, image_token_id=IMG, boi_token_id=BOI, eoi_token_id=EOI)
    c._attn_implementation = impl
    torch.manual_seed(0)
    return Gemma4ForConditionalGeneration(c).eval()


@pytest.fixture(scope="module")
def batch():
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
    ims = [Image.new("RGB", (64, 48), (200, 10, 10)), Image.new("RGB", (40, 40), (10, 200, 90))]
    vis = Gemma4ImageProcessor()(images=ims, return_tensors="pt", max_soft_tokens=70)
    rng = random.Random(0)
    unpacked = []
    for i in range(2):
        pre = [2, 5, BOI] + [IMG] * int(vis["num_soft_tokens_per_image"][i]) + [EOI, 7, 8]
        unpacked.append([pre + [rng.randrange(3, 50) for _ in range(rng.randrange(3, 14))] for _ in range(4)])
    return vis, unpacked


def unpacked_logits(m, vis, unpacked):
    out = []
    for i, seqs in enumerate(unpacked):
        for s in seqs:
            x = torch.tensor([s])
            o = m(input_ids=x, pixel_values=vis["pixel_values"][i:i + 1], mm_token_type_ids=(x == IMG).long(),
                  image_position_ids=vis["image_position_ids"][i:i + 1])
            out.append(o.logits[0, -1])
    return torch.stack(out)


def packed_logits(m, vis, unpacked, flat_positions=False, bidirectional=False):
    ids, block, pos, _, ro = collate_packed([pack(s, min_prefix=60) for s in unpacked])
    if flat_positions:
        pos = torch.arange(ids.shape[1])[None].expand_as(ids)
    return packed_readout_logits(m, ids, block, pos, ro, bidirectional=bidirectional,
                                 pixel_values=vis["pixel_values"], image_position_ids=vis["image_position_ids"])


@pytest.mark.parametrize("bidir,impl", [(None, "sdpa"), ("vision", "sdpa"), (None, "eager")])
@torch.no_grad()
def test_packed_equals_unpacked(batch, bidir, impl):
    vis, unpacked = batch
    m = tiny_gemma4(bidir, window=8, impl=impl)          # window 8 ≪ ~70 image tokens: sliding layers are exercised
    ref = unpacked_logits(m, vis, unpacked)
    got = packed_logits(m, vis, unpacked)
    assert got.shape == ref.shape == (8, 64)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)
    # controls: flat positions (no per-block restart) and the trained bidirectional mask both change the logits.
    # The bidirectional one needs a model without KV sharing: here the only global layer that computes its own K/V
    # sits below the shared ones, so the readout token (which already sees its whole block) cannot notice.
    assert (packed_logits(m, vis, unpacked, flat_positions=True) - ref).abs().max() > 1e-3
    m0 = tiny_gemma4(bidir, window=8, impl=impl, kv_shared=0)
    assert (packed_logits(m0, vis, unpacked, bidirectional=True) - unpacked_logits(m0, vis, unpacked)).abs().max() > 1e-3


def test_pack_and_masks():
    ids, block, pos, ro = pack([[1, 2, 3, 4], [1, 2, 3, 5, 6], [1, 2, 7]])
    assert ids == [1, 2, 3, 4, 3, 5, 6, 7] and block == [0, 0, 1, 1, 2, 2, 2, 3]
    assert pos == [0, 1, 2, 3, 2, 3, 4, 2] and ro == [3, 6, 7]
    with pytest.raises(ValueError):
        pack([[1, 2, 3], [1, 9, 3]], min_prefix=2)
    ids, block, pos, valid, _ = collate_packed([(ids, block, pos, ro)])
    m = block_masks(block, pos, sliding_window=2)
    full, sl = m["full_attention"][0, 0], m["sliding_attention"][0, 0]
    assert full[4, 2] == 0 and full[4, 1] == 1                # block 2 never sees block 1, always sees the prefix
    assert full[1, 2] == 0 and full[4, 5] == 0                # the prefix never sees a block; causal inside a block
    assert sl[6, 1] == 0 and sl[6, 5] == 1                    # window 2 measured in per-block positions (4 − 1 = 3)
    mb = block_masks(block, pos, sliding_window=2, bidirectional=True)["full_attention"][0, 0]
    assert mb[4, 6] == 1 and mb[4, 2] == 0 and mb[0, 1] == 1  # bidirectional inside a block / the prefix only


def test_packed_gradients_flow(batch):
    vis, unpacked = batch
    m = tiny_gemma4().train()
    packed_logits(m, vis, unpacked, bidirectional=True).logsumexp(-1).sum().backward()
    assert m.model.language_model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_letter_readout_pads_member_to_vocab():
    """The letter-id table stops at the largest letter token id; logits span the whole vocabulary."""
    import torch.nn as nn
    from bio_waila.models.jev_gemma import GemmaBackend
    be = GemmaBackend.__new__(GemmaBackend)
    nn.Module.__init__(be)
    member = torch.zeros(26, 5, dtype=torch.bool)
    member[0, 3] = member[1, 4] = True
    be.register_buffer("member", member, persistent=False)
    logits = torch.zeros(2, 9)
    logits[:, 3] = 5.0
    lp = be._letters(logits)
    assert lp.shape == (2, 26) and lp[0, 0] > lp[0, 1] and torch.isinf(lp[0, 2])
