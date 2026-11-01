"""F4: all of an image's field questions in ONE sequence over a shared image prefix (mammoth-1 layout,
docs/models/baseten_decision_recipe.md §4).

    unpacked (F2)   [prefix | q_modality]   [prefix | q_stain]   ...        one forward per (image, field)
    packed   (F4)   [prefix | q_modality | q_stain | ...]                   one forward per image

`prefix` is the longest common token prefix of the image's unpacked prompts (chat header, image soft tokens and
the shared "Field: " text). Every question block restarts its position ids at len(prefix), so each block sees the
exact token positions of its own unpacked sequence. Masks are built per Gemma-4 layer type and passed as a dict,
which Gemma4Model hands straight to its decoder layers:

    structure   a token sees its own block and the prefix; the prefix never sees a block; blocks never see each other
    causal      key position ≤ query position (per-block positions)
    sliding     query position − key position < sliding_window  (sliding layers only; per-block positions, so a late
                block keeps the image in view exactly as its unpacked sequence would)
    vision      config.use_bidirectional_attention == "vision" (26B-A4B): image tokens of one image see each other
                in sliding layers, as Gemma4's own create_masks_for_vision_model does
    bidir       optional (`bidirectional=True`, the trained variant): inside the prefix and inside each block, global
                layers attend in both directions; sliding layers stay causal (mammoth-1 kept its linear layers causal)

With `bidirectional=False` the readout logits equal the unpacked F2 logits up to float error
(tests/test_jev_packed.py, scripts/check_f4_equivalence.py).
"""
import torch


def pack(seqs, min_prefix=0):
    """Pack one image's unpacked token sequences (lists of ids sharing a prefix).

    Returns (ids, block, pos, readout): token ids, block id per token (0 = prefix, j + 1 = question j), position
    id per token, and the index of the last token of each question block (where the answer letter is read).
    The prefix is the longest common prefix, capped so every block keeps at least one token."""
    p = min(len(s) for s in seqs) - 1
    for i in range(p):
        if any(s[i] != seqs[0][i] for s in seqs[1:]):
            p = i
            break
    if p < min_prefix:
        raise ValueError(f"shared prefix ({p} tokens) ends before the image ({min_prefix} tokens)")
    ids, block, pos, readout = list(seqs[0][:p]), [0] * p, list(range(p)), []
    for j, s in enumerate(seqs):
        suf = s[p:]
        ids += suf
        block += [j + 1] * len(suf)
        pos += range(p, p + len(suf))
        readout.append(len(ids) - 1)
    return ids, block, pos, readout


def collate_packed(packed, pad_id=0):
    """Right-pad a batch of `pack` outputs → ids, block (−1 on padding), pos, valid [B, T] and readouts."""
    T = max(len(x[0]) for x in packed)
    B = len(packed)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    block = torch.full((B, T), -1, dtype=torch.long)
    pos = torch.zeros((B, T), dtype=torch.long)
    for b, (i, bl, ps, _) in enumerate(packed):
        n = len(i)
        ids[b, :n], block[b, :n], pos[b, :n] = torch.tensor(i), torch.tensor(bl), torch.tensor(ps)
    return ids, block, pos, block >= 0, [x[3] for x in packed]


def block_masks(block, pos, sliding_window, vision_group=None, bidirectional=False):
    """{"full_attention", "sliding_attention"}: bool [B, 1, T, T], True = attend.

    block [B, T] (0 prefix, j > 0 question j, −1 padding), pos [B, T] per-block positions, vision_group [B, T]
    (image index for image soft tokens, −1 elsewhere) or None when the model has no vision bidirectionality."""
    bq, bk = block[:, :, None], block[:, None, :]
    pq, pk = pos[:, :, None], pos[:, None, :]
    valid_k = bk >= 0
    structure = valid_k & ((bk == bq) | (bk == 0)) & (bq >= 0)
    causal = pk <= pq
    same_block = bk == bq
    full = structure & (causal | same_block) if bidirectional else structure & causal
    local = causal
    if vision_group is not None:
        vq, vk = vision_group[:, :, None], vision_group[:, None, :]
        local = local | ((vq >= 0) & (vq == vk))
    sliding = structure & local & (pk > pq - sliding_window)
    eye = torch.eye(block.shape[1], dtype=torch.bool, device=block.device)[None]
    pad_q = (block < 0)[:, :, None]
    full, sliding = full | (eye & pad_q), sliding | (eye & pad_q)      # padding rows see themselves (no NaN rows)
    return {"full_attention": full[:, None], "sliding_attention": sliding[:, None]}


def as_attn_masks(masks, impl, dtype):
    """Bool masks → what the attention implementation takes (sdpa: bool; eager: additive float)."""
    if impl == "eager":
        neg = torch.finfo(dtype).min
        return {k: torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill(~m, neg) for k, m in masks.items()}
    if impl not in ("sdpa", None):
        raise ValueError(f"F4 needs sdpa or eager attention (dense 4D masks), not {impl}")
    return masks


def vision_groups(ids, block, image_token_id):
    """Image index per image soft token (−1 elsewhere): runs of image tokens inside the prefix."""
    is_img = (ids == image_token_id) & (block == 0)
    prev = torch.roll(is_img, 1, dims=-1)
    prev[..., 0] = False
    starts = is_img & ~prev
    return torch.where(is_img, torch.cumsum(starts.long(), -1) - 1, torch.full_like(ids, -1))


def packed_readout_logits(cg_model, ids, block, pos, readouts, bidirectional=False, **vision_inputs):
    """Final logits [sum of blocks, V] at every readout position of a packed batch.

    `cg_model` is a Gemma4ForConditionalGeneration (PEFT-wrapped is fine: LoRA layers sit inside its modules);
    `vision_inputs` are the processor's image tensors for the B images (pixel_values, image_position_ids)."""
    cfg = cg_model.config
    tcfg = cfg.get_text_config()
    vg = vision_groups(ids, block, cfg.image_token_id) if tcfg.use_bidirectional_attention == "vision" else None
    masks = block_masks(block, pos, tcfg.sliding_window, vg, bidirectional)
    dtype = next(cg_model.parameters()).dtype
    masks = as_attn_masks(masks, cfg._attn_implementation, dtype)
    out = cg_model.model(input_ids=ids, attention_mask=masks, position_ids=pos, use_cache=False, **vision_inputs)
    h = out.last_hidden_state
    b_idx = torch.cat([torch.full((len(r),), b, dtype=torch.long) for b, r in enumerate(readouts)]).to(h.device)
    t_idx = torch.tensor([t for r in readouts for t in r], dtype=torch.long, device=h.device)
    logits = cg_model.lm_head(h[b_idx, t_idx])
    if (cap := tcfg.final_logit_softcapping) is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits
