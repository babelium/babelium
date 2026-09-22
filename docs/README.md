# Docs

Rewritten, English-language documentation of the token theory and the research that informs it. Source material lives in `context/` (not tracked in this repo); everything here is a standalone rewrite for publication, not a direct copy.

## The theory

- [token-antinomy.md](token-antinomy.md) — the core essay. A token is a conversion interface between discrete identity and continuous similarity, not a slice of a signal. Five case studies (autoregressive sampling, vector quantization, diffusion/AlphaFold2, retrieval, absent-address slots), a unified notation, a displacement criterion for measuring information loss, and a real internal conflict resolved by token healing.
- [theory-signature.md](theory-signature.md) — the formal ground layer: what structure each face (identity, similarity) actually needs, the quotient construction behind identity, the anchor-and-comparison construction behind similarity, and the seven-part conversion signature.
- [theory-criterion.md](theory-criterion.md) — the displacement criterion in full: Phi declared from architecture, the zero-kernel theorem, the two-term cost (quotient collapse, selection collapse), and the section-property conflict resolved via token healing.
- [theory-mechanisms.md](theory-mechanisms.md) — the composition calculus: three assembly operators, five inference rules, and four derivations (why AR and diffusion have different per-step cost, why attention sinks form, why some tokens are unreachable, why MLA's cost can be lower than a standard baseline).

## Survey background

- [survey-cross-modal-tokenization.md](survey-cross-modal-tokenization.md) — how tokenization evolved in text, vision, and 3D mesh, and the regularities that recur across all three.
- [survey-formal-tokenization-theory.md](survey-formal-tokenization-theory.md) — three papers putting tokenization on formal footing: round-trip consistency, coding efficiency, and character-level surprisal.
- [survey-dynamic-boundaries.md](survey-dynamic-boundaries.md) — BLT (entropy-based patch boundaries) and H-Net (learned, end-to-end chunking).
- [survey-attention-sinks.md](survey-attention-sinks.md) — four papers on why attention sinks form, a mechanistic circuit for the P0 case, the dissociation from massive activations, and what survives without autoregression.
- [survey-structural-slots.md](survey-structural-slots.md) — what a sink actually carries once formed (Catch-Tag-Release), and that the same repurposed-slot phenomenon appears in vision transformers with no causal mask at all (ViT Registers).
- [survey-evidence-chain.md](survey-evidence-chain.md) — a synthesis of the boundary-formation evidence across all seven papers above: what is and is not supported about commitment, uncertainty, and structure as boundary drivers.
- [survey-nearest-prior-work.md](survey-nearest-prior-work.md) — a prior-work check against 2026 KV-cache-quantization literature, locating where the displacement criterion's core idea was already derived independently, and what survives as a genuine increment.
