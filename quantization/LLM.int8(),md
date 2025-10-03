### Paragraph: **Abstract** — one-sentence paraphrase

The paper introduces **LLM.int8()**, a two-part int8 quantization procedure (vector-wise quantization + mixed-precision decomposition) that lets you run multi-billion-parameter transformers (up to 175B) in ~8-bit for almost all computations while preserving full fp16 accuracy and halving memory. 

### Plain English explanation (undergrad ML level)

* **Problem:** Large transformers require huge GPU memory. Converting all weights to low precision (int8) saves memory but usually hurts performance — especially once models scale beyond a few billion parameters.
* **Key insights:** Transformers develop **systematic, sparse outlier features** (a tiny fraction of dimensions have very large magnitudes) that destroy conventional quantization.
* **Solution (LLM.int8()):**

  1. **Vector-wise quantization**: use different normalization constants per inner product (row/column) so small and medium features are quantized with good local resolution.
  2. **Mixed-precision decomposition**: isolate those rare outlier feature dimensions and compute them in **fp16**, while the rest (>99.9%) use **int8** multiplication.
* **Effect:** Most of the model runs in fast/compact int8; a tiny fp16 correction preserves accuracy. This yields ~2× memory reduction with no performance degradation. The implementation and experiments (up to 175B) back this up. 

### Math / formulas (direct & derived)

1. **Absmax (symmetric) quantization** used as a base: for an FP16 tensor (X_{f16}),
   [
   X_{i8} ;=; \left\lfloor \frac{127}{|X_{f16}|*\infty}; X*{f16} \right\rceil
   ]
   so the scale is (s_X = \frac{127}{|X_{f16}|_\infty}). (Notation: (\lfloor\cdot\rceil) = round to nearest integer.) 

2. **Matrix multiplication decomposition** (core idea): given input activations (X) and weights (W),
   [
   Y ;=; XW.
   ]
   Decompose (W) into an 8-bit part and an outlier (fp16) part:
   [
   W ;=; W_{\text{int8}} + W_{\text{outlier}},
   ]
   so
   [
   Y ;=; XW_{\text{int8}} ;+; XW_{\text{outlier}}.
   ]
   The first term is performed with int8→int32 accumulators then dequantized; the second term is the small fp16 correction. (Schematic in paper Figure 2.) 

3. **Vector-wise quantization** (brief formulaic view): Treat the matrix multiply as many inner products and have scaling constants per row of (X) and per column of (W). Let (c_x) be row scales and (c_w) column scales; quantize
   [
   A_{i8} = Q(X) \quad,\quad B_{i8} = Q(W)
   ]
   and after int8 matmul produce int32 outputs (O_{i32}) which are dequantized by elementwise dividing with the outer product (c_x \otimes c_w):
   [
   Y \approx \frac{1}{c_x \otimes c_w}; O_{i32}.
   ]
   (See Section 3 / eqns in the paper for the precise notation.) 

---

### Check question (one short question)

**Why do the authors isolate only a tiny subset of feature dimensions into fp16 instead of keeping everything in fp16?**

Reply in one short sentence. (I’ll follow the two-attempts rule: if your first answer misses something, I’ll give a hint; second miss → I’ll show the correct reasoning.)


Medical References:
1. None — DOI: file-1s6Xq47tRZ3RXkHfSwEv6P
2. None — DOI: file-4qfX2vmzpMVSbrU5gCmJuy
