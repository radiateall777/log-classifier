import torch


def build_special_token_mask(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Build mask for special tokens.

    Special tokens include:
        pad / cls / sep / bos / eos / unk 等 tokenizer.all_special_ids 中的 token.

    Args:
        input_ids: [B, L]
        tokenizer: HuggingFace tokenizer

    Returns:
        special_mask: [B, L], True means this position is special token.
    """
    special_ids = set(tokenizer.all_special_ids)

    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        special_mask |= input_ids.eq(int(token_id))

    return special_mask


def apply_unk_token_noise(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    noise_prob: float,
    min_keep_tokens: int = 1,
    generator: torch.Generator | None = None,
):
    """
    Fixed-ratio input-id-level UNK replacement.

    This function implements the second robustness definition:

        1. Tokenize text first.
        2. Directly replace content token ids with tokenizer.unk_token_id.
        3. Do not decode.
        4. Do not retokenize.
        5. Keep attention_mask unchanged.

    For each sample, it replaces:

        int(num_valid_content_tokens * noise_prob)

    valid content tokens with UNK.

    Special tokens and padding tokens are never replaced.

    Args:
        input_ids:
            Tensor [B, L]
        attention_mask:
            Tensor [B, L]
        tokenizer:
            HuggingFace tokenizer
        noise_prob:
            Missing ratio in [0, 1].
        min_keep_tokens:
            Minimum number of valid content tokens to preserve per sample.
        generator:
            Optional torch.Generator for deterministic sampling.

    Returns:
        noisy_input_ids:
            Tensor [B, L]
        noise_mask:
            Bool Tensor [B, L], True means this position was replaced by UNK.
    """
    if tokenizer.unk_token_id is None:
        raise ValueError(
            "tokenizer.unk_token_id is None. "
            "UNK-token noise requires a tokenizer with unk_token_id."
        )

    if not 0.0 <= float(noise_prob) <= 1.0:
        raise ValueError(f"noise_prob must be in [0, 1], got {noise_prob}")

    noisy_input_ids = input_ids.clone()
    device = input_ids.device

    special_mask = build_special_token_mask(input_ids, tokenizer).to(device)
    valid_mask = attention_mask.bool() & (~special_mask)

    noise_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    batch_size = input_ids.size(0)

    for i in range(batch_size):
        valid_positions = valid_mask[i].nonzero(as_tuple=True)[0]
        num_valid = valid_positions.numel()

        if num_valid == 0:
            continue

        if num_valid <= int(min_keep_tokens):
            continue

        max_maskable = max(0, num_valid - int(min_keep_tokens))
        n_mask = int(num_valid * float(noise_prob))
        n_mask = min(n_mask, max_maskable)

        if n_mask <= 0:
            continue

        perm = torch.randperm(
            num_valid,
            device=device,
            generator=generator,
        )
        selected_positions = valid_positions[perm[:n_mask]]
        noise_mask[i, selected_positions] = True

    noisy_input_ids[noise_mask] = int(tokenizer.unk_token_id)

    return noisy_input_ids, noise_mask