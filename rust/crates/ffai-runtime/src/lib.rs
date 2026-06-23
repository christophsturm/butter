// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0

//! # ffai-runtime
//!
//! Generation orchestration: logits selection, sampling, and the
//! prefill→decode loop. **Pure CPU logic, zero GPU/backend dependency** — the
//! model passes a `step(token, pos) -> logits` closure (which owns its KV cache
//! and runs on whatever [`ffai_core::Device`](../ffai_core) it likes), and this
//! drives it. So the generation runtime is written ONCE and shared across every
//! backend, every model, and (via FFI) the Swift host.

/// Index of the max logit.
pub fn argmax(logits: &[f32]) -> usize {
    (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap_or(0)
}

/// The `k` highest-logit indices, descending.
pub fn topk(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    idx.truncate(k);
    idx
}

/// How to turn logits into the next token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    /// argmax — deterministic.
    Greedy,
    /// softmax(logits / temperature), sample.
    Temperature(f32),
    /// temperature + keep only the top-`k` logits.
    TopK(f32, usize),
    /// temperature + nucleus: smallest set whose cumulative prob ≥ `p`.
    TopP(f32, f32),
}

/// Tiny dep-free PRNG (SplitMix64) so sampling is reproducible without pulling
/// in `rand`. `next_f32` yields a uniform value in `[0, 1)`.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32) / (1u32 << 24) as f32
    }
}

/// Pick the next token id from `logits` under `s`.
pub fn sample(logits: &[f32], s: &Sampling, rng: &mut Rng) -> usize {
    let temp = match *s {
        Sampling::Greedy => return argmax(logits),
        Sampling::Temperature(t) | Sampling::TopK(t, _) | Sampling::TopP(t, _) => t,
    };
    if temp <= 0.0 {
        return argmax(logits);
    }
    let mut cand: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    match *s {
        Sampling::TopK(_, k) => cand.truncate(k.max(1)),
        Sampling::TopP(_, p) => {
            let mx = cand[0].1;
            let exps: Vec<f32> = cand.iter().map(|&(_, l)| ((l - mx) / temp).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let mut cum = 0.0;
            let mut keep = 0;
            for (i, &e) in exps.iter().enumerate() {
                cum += e / sum;
                keep = i + 1;
                if cum >= p {
                    break;
                }
            }
            cand.truncate(keep.max(1));
        }
        _ => {}
    }
    let mx = cand[0].1;
    let exps: Vec<f32> = cand.iter().map(|&(_, l)| ((l - mx) / temp).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let r = rng.next_f32() * sum;
    let mut cum = 0.0;
    for (k, &e) in exps.iter().enumerate() {
        cum += e;
        if r < cum {
            return cand[k].0;
        }
    }
    cand[cand.len() - 1].0
}

/// When to stop decoding.
#[derive(Debug, Clone)]
pub struct StopOn {
    pub max_new: usize,
    pub eos: Option<u32>,
}

/// Drive a model: prefill the prompt, then sample-and-step until `stop`.
///
/// `step(token, pos) -> logits` runs one model step (the model owns its KV
/// cache + device). Returns the generated token ids (prompt not included).
/// Backend- and model-agnostic — the whole generation loop, shared.
pub fn generate(
    prompt: &[u32],
    stop: &StopOn,
    sampling: &Sampling,
    seed: u64,
    mut step: impl FnMut(u32, usize) -> Vec<f32>,
) -> Vec<u32> {
    assert!(!prompt.is_empty(), "generate: empty prompt");
    let mut logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        logits = step(tok, pos);
    }
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(stop.max_new);
    let mut pos = prompt.len();
    for _ in 0..stop.max_new {
        let next = sample(&logits, sampling, &mut rng) as u32;
        out.push(next);
        if Some(next) == stop.eos {
            break;
        }
        logits = step(next, pos);
        pos += 1;
    }
    out
}

/// Legacy decoding params kept for the FFI/skeleton surface.
#[derive(Debug, Clone)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}
impl Default for SampleParams {
    fn default() -> Self {
        SampleParams { temperature: 0.7, top_p: 0.95, max_tokens: 256 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_topk() {
        let l = [0.1, 0.5, 0.2, 0.9, 0.3];
        assert_eq!(argmax(&l), 3);
        assert_eq!(topk(&l, 3), vec![3, 1, 4]);
    }

    #[test]
    fn greedy_is_argmax() {
        let l = [1.0, 3.0, 2.0];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&l, &Sampling::Greedy, &mut rng), 1);
    }

    #[test]
    fn generate_greedy_drives_a_stepper() {
        // toy "model": logits peak at (pos+1) % vocab ⇒ greedy emits a ramp.
        let vocab = 8;
        let out = generate(&[0], &StopOn { max_new: 4, eos: None }, &Sampling::Greedy, 0, |_t, pos| {
            let mut l = vec![0.0f32; vocab];
            l[(pos + 1) % vocab] = 1.0;
            l
        });
        assert_eq!(out, vec![1, 2, 3, 4]);
    }

    #[test]
    fn topp_collapses_to_dominant_logit() {
        let l = [10.0, 0.0, 0.0, 0.0];
        let mut rng = Rng::new(42);
        for _ in 0..20 {
            assert_eq!(sample(&l, &Sampling::TopP(1.0, 0.9), &mut rng), 0);
        }
    }
}
