# Future Development and Research Roadmap

This file collects architectural and systems ideas that are **not necessarily implemented yet**.

Its purpose is to keep future research separate from the current-state README. Ideas can be added here early, refined as experiments become clearer, and moved into implementation/history documentation once they are validated.

The ordering below is not a strict schedule. The project should remain empirical: an idea moves forward because measurements justify it, not because it appears elegant on paper.

---

# 1. Research philosophy

The project is built around a deliberate constraint:

> **The model should remain locally trainable and usable on consumer-class hardware.**

That constraint should continue to influence architecture decisions.

The easiest answer to many LLM problems is to add:

- more parameters;
- more context tokens;
- more GPUs;
- more training tokens;
- more memory bandwidth.

Those approaches are valid, but they can hide inefficient formulations. This project should repeatedly ask the complementary question:

> Can the same capability be obtained with a better representation, a better routing decision, a cheaper numerical formulation, or a more useful allocation of the available compute?

Scaling is not forbidden. It should be used when the smaller architecture has learned enough to justify it.

---

# 2. Current baseline to preserve

Future experiments should remain comparable to the current routed-prefix model.

The present reference architecture has:

```text
~501M stored parameters
12 Transformer layers
768 model width
6 MoE experts / top-2 execution
4k continuous current window
up to ~60k external searchable same-document history
128-token historical blocks
up to 16 globally routed blocks
up to 2k exact historical-token prefix
maximum ~6k deep active sequence
```

The deep Transformer also uses a mixture of:

```text
local attention
+ dilated attention
+ deterministic global sparse attention
+ learned retrieval attention
```

This is important: **sparse context construction and sparse attention are separate concepts**.

The top-level context router decides which old tokens deserve to become part of persistent working context. Sparse attention decides which interactions are worth computing inside or alongside the active Transformer state.

---

# 3. Immediate evaluation priorities

Before making the memory architecture substantially more complicated, establish whether the current router works.

## 3.1 Router ablations

Evaluate the same checkpoint under:

```text
A. current 4k only
B. most-recent historical prefix + current 4k
C. random historical blocks + current 4k
D. learned routed historical blocks + current 4k
```

Metrics should include both LM loss and targeted long-context tasks.

The key question is not simply whether adding 2k tokens improves performance. It is whether **learned selection beats cheap alternatives**.

## 3.2 Long-range dependency suite

Construct controlled examples where required information appears at known distances such as:

```text
8k
16k
32k
48k
60k
```

Test:

- whether the correct block is selected;
- whether retrieval changes the output probability of the correct answer;
- whether success degrades with source distance;
- whether the model actually uses retrieved information after selection.

## 3.3 Router diagnostics

Track aggregated statistics rather than only individual diagnostic snapshots:

- fraction of plain-current examples;
- fraction using direct partial history;
- fraction requiring learned routing;
- candidate block count;
- unique selected blocks;
- selection entropy;
- selected source distance;
- overlap with recent-history baseline;
- block reuse frequency;
- router score margin around the top-k boundary.

## 3.4 Standard language-model evaluation

Maintain a stable evaluation suite including:

- held-out mixed-corpus loss;
- WikiText-2 loss/perplexity;
- fixed qualitative prompts;
- basic coding benchmarks;
- scientific/technical completion tests.

Record loss in nats/token in addition to perplexity when comparing models using different tokenizers.

---

# 4. Improve the global context router

The current router is deliberately simple. It is useful as a baseline because success or failure can be interpreted easily.

Current high-level form:

```text
current 4k embeddings
      ↓ mean/query projection
small router query
      ↓
mean-projected historical blocks
      ↓
dot-product scores
      ↓
hard top-k
```

Possible extensions should be introduced one at a time.

## 4.1 Better current-context representation

Replace simple mean pooling with a cheap semantic encoder, for example:

```text
current 4k embeddings
      ↓
small 1–2 layer router encoder
      ↓
64–128 dimensional context query
```

The router should remain far cheaper than the Transformer trunk.

## 4.2 Better historical block representations

Candidate ideas:

- learned pooling within each 128-token block;
- first/last/mean mixtures;
- small block encoder;
- multiple keys per block;
- semantic and lexical keys in parallel;
- keys created from representations captured when tokens leave the active window.

The benefit must justify both compute and memory cost.

## 4.3 Multi-scale blocks

Some dependencies are naturally local while others correspond to an entire section.

Possible hierarchy:

```text
128-token fine blocks
512-token medium blocks
2048-token coarse blocks
```

A coarse pass could identify relevant regions before fine selection.

This could make very long searchable horizons practical without evaluating thousands of fine blocks equally.

## 4.4 Adaptive retrieval budget

The current model has a fixed maximum of 16 routed blocks.

Future experiments could allow the router to choose a smaller/larger budget based on confidence or context complexity, while preserving a strict compute ceiling.

Possible strategies:

- score threshold plus maximum K;
- learned stop probability;
- entropy-dependent budget;
- fixed small/medium/large routing modes.

Any adaptive scheme must avoid leaking information through artificial ordering or budget encoding.

---

# 5. Sparse context creation + residual sparse attention

This is a particularly promising future direction.

The key idea is **not** to make sparse attention operate only inside the globally selected context. Instead, give the global router and per-layer sparse retrieval different responsibilities.

## 5.1 Global context = persistent working memory

The top-level router selects information broadly relevant to the whole current section:

```text
full historical store
       ↓
global context router
       ↓
important old blocks
       ↓
exact historical tokens
       ↓
[persistent routed prefix | current 4k]
       ↓
all Transformer layers
```

These selected historical tokens become part of the model's active reasoning state and are processed through the complete Transformer stack.

Conceptually:

> **What historical information should I actively remember while processing this current window?**

## 5.2 Residual sparse attention = missing information

At each layer, sparse-attention retrieval can additionally query the historical store for information that was **not** included in the persistent context.

```text
                         full history
                              │
              ┌───────────────┴────────────────┐
              │                                │
              ▼                                ▼
      global context router          residual history pool
              │                                │
      persistent exact prefix                  │
              │                                │
              └──── current Transformer ───────┘
                         │
                  layer-specific query
                         │
                sparse residual access
```

The residual search set should explicitly exclude blocks already inserted by the global context router:

\[
\mathcal{M}_{\mathrm{residual}}
=
\mathcal{M}_{\mathrm{history}}
\setminus
\mathcal{M}_{\mathrm{global}}.
\]

This prevents sparse attention from wasting its limited retrieval budget rediscovering information already present in working context.

The semantic division becomes:

```text
global routed context:
    "What should I keep in working memory?"

residual sparse attention:
    "What additional information am I missing at this layer/query?"
```

## 5.3 Why this combination is attractive

Unlike a two-stage filtering hierarchy, residual sparse attention provides an **escape path** when the global router misses something.

If a useful block is omitted from persistent context, a later layer can still retrieve it from the residual history.

This makes the two mechanisms complementary rather than multiplicatively fragile.

It also permits different depths to retrieve different information:

```text
early layers   -> lexical / identifier / syntactic details
middle layers  -> relations / definitions / equations
later layers   -> higher-level semantic or argumentative dependencies
```

## 5.4 Query granularity

Do not initially route independently for every one of the 4096 current tokens.

A cheaper first design could use query groups, for example:

```text
4k current context
      ↓
32 × 128-token query chunks
      ↓
1 small residual-memory query per chunk/layer
      ↓
top-1..4 residual blocks
```

This keeps routing overhead bounded while allowing different portions of the current context to request different information.

## 5.5 Residual-memory representation problem

A retrieved historical block at Transformer layer 8 does not automatically possess a layer-8 hidden representation because it was not processed through layers 0–7 as part of the current active context.

Candidate solutions:

### Static projected memory

Store a compact representation derived from raw token embeddings and let each layer project it into its own K/V space.

Pros:

- simple;
- cheap;
- bounded memory.

Cons:

- less semantically mature than true layer-specific hidden states.

### Small memory encoder

Process historical blocks once through a shallow independent encoder.

Pros:

- richer semantic representation;
- still much cheaper than the full Transformer.

Cons:

- additional parameters and training coupling.

### Eviction-time memory

During autoregressive inference, save a compressed representation when tokens leave the active 4k window after they have already passed through the model.

Pros:

- memory is derived from contextualized representations;
- attractive interpretation as learned episodic memory.

Cons:

- training equivalence is harder;
- representation versioning across layers needs careful design.

This representation question should be resolved experimentally before adding a large residual-memory system.

## 5.6 Training considerations

The global and residual routers should receive different incentives naturally:

```text
global router
    useful across many current tokens / many layers

residual sparse retrieval
    useful to a particular layer or query group
```

Potential training safeguards:

- occasional random/distractor context blocks;
- residual-retrieval dropout;
- controlled masking of global blocks during diagnostic training;
- explicit measurement of whether residual attention recovers global-router misses;
- avoid auxiliary objectives unless the LM loss is insufficient.

The primary objective should remain next-token prediction whenever possible.

---

# 6. Longer addressable memory without larger deep context

If sparse context works, extending the searchable horizon should not require linearly increasing deep Transformer length.

A possible progression is:

```text
64k   searchable -> ~6k active
128k  searchable -> bounded active context
256k  searchable -> bounded active context
1M    searchable -> hierarchical/coarse-to-fine router
```

The desired asymptotic structure is:

\[
N_{\mathrm{history}}
\gg
N_{\mathrm{active}}
\gg
N_{\mathrm{attended\ per\ query}}.
\]

For example:

```text
1,000,000 addressable tokens
        ↓ cheap context retrieval
16,000 active tokens
        ↓ sparse residual attention
~2,000 effective interactions per query
```

This is a more promising local-compute target than attempting dense million-token attention.

---

# 7. Adaptive training compute: progressive depth and token learning dynamics

The largest remaining practical constraint is no longer only model memory or kernel efficiency. Even a well-optimized local model requires a very large amount of wall-clock time to consume tens or hundreds of billions of useful training tokens.

A major research direction should therefore ask a more fundamental question:

> **Does every token need to be processed by the full final model, and does every token need to contribute equally throughout training?**

Two complementary ideas should be investigated in sequence:

1. **progressive residual depth growth**, so early tokens are processed by a much smaller model and new Transformer blocks are introduced only when additional capacity becomes useful;
2. **online token learning dynamics**, so later training increasingly concentrates gradient and output-head compute on tokens that are still informative while retaining random exploration.

These mechanisms should be tested separately before they are combined. The first experiment should modify only model depth. Token selection should be introduced only after the progressive-depth behavior is understood.

## 7.1 Stage 1: progressive residual Transformer growth

Instead of constructing the final 12-layer model from the first training step, begin with a very shallow model and increase depth progressively:

```text
1 block
   ↓
2 blocks
   ↓
3 blocks
   ↓
...
   ↓
12 blocks
```

The motivation is simple: a large amount of early language learning may not require the full final depth. A one- or two-block model can potentially learn frequent token statistics, lexical structure, punctuation, syntax, common code patterns, and broad semantic regularities at much higher token throughput than a 12-block model.

The final blocks should only begin consuming compute once the shallower model is no longer using its available compute efficiently.

The primary quantity to optimize is therefore not nominal tokens seen but something closer to:

\[
\text{validation improvement per unit compute}.
\]

The central experiment is whether progressive depth improves:

\[
L_{\mathrm{val}}(C),
\]

where \(C\) is total training compute or wall-clock cost, relative to conventional fixed-depth training.

## 7.2 Exact identity insertion through learned residual gates

New Transformer blocks must be insertable without perturbing the function already learned by the current model.

For a pre-norm Transformer block, use independent learned residual scalars for the attention and MLP/MoE branches:

\[
y_l
=
x_l
+
\alpha_{A,l}\,
A_l\!\left(N_l(x_l)\right),
\]

\[
x_{l+1}
=
y_l
+
\alpha_{M,l}\,
M_l\!\left(N'_l(y_l)\right).
\]

For an existing active block, the alphas behave as ordinary trainable parameters. For a newly inserted block initialize:

\[
\boxed{
\alpha_{A,l}=0,
\qquad
\alpha_{M,l}=0
}
\]

while initializing the internal Transformer parameters normally and independently at random.

At insertion time:

\[
y_l=x_l,
\qquad
x_{l+1}=x_l,
\]

so the new block is an **exact identity** even though its attention, MoE, and projection weights are randomly initialized.

This avoids forcing the attention mechanism itself to behave like an identity. The residual path already transports the input exactly; the new branch only needs to begin with zero contribution.

The alpha multiplication should be fused into the existing residual-add path:

```text
y = x + alpha * residual
```

rather than implemented as a separate tensor-scale kernel. The extra scalar multiply is negligible relative to QKV, attention, MoE, and MLP computation.

## 7.3 Random initialization is intentional

Do **not** initialize a newly inserted block by copying an existing trained block in the first experiment.

Copying could make the new block strongly correlated with the transformation already represented by neighboring layers and bias optimization toward a redundant local representation.

Instead use:

```text
standard random Transformer initialization
+
alpha_attention = 0
+
alpha_mlp = 0
```

This gives both desired properties:

```text
function preservation at insertion
+
new independent degrees of freedom
```

The new block contributes nothing until optimization finds a useful direction in which to open its residual gates.

For

\[
x'=x+\alpha F(x;\theta),
\]

at exactly \(\alpha=0\), the internal branch parameters initially receive no gradient:

\[
\frac{\partial L}{\partial \theta}
=
\alpha
\frac{\partial L}{\partial F}
\frac{\partial F}{\partial \theta}
=0,
\]

while the gate itself generally receives a nonzero gradient:

\[
\frac{\partial L}{\partial \alpha}
=
\frac{\partial L}{\partial x'}\cdot F(x;\theta).
\]

The resulting natural sequence is:

```text
new random block inserted
        ↓
alpha = 0
        ↓
alpha learns whether the branch is useful
        ↓
alpha moves away from zero
        ↓
internal block parameters begin receiving gradient
        ↓
new and old blocks co-adapt
```

The model should be allowed to learn the alpha values directly. Do not impose a hand-designed opening schedule in the first experiment. Do not initially clamp alpha to \([0,1]\); a negative or greater-than-one value may be a legitimate learned residual scaling. Alpha parameters should normally be excluded from weight decay.

## 7.4 Previously trained layers remain trainable

Progressive depth should **not** mean freezing the earlier model and permanently fitting only residual corrections on top of it.

A shallow network is forced to learn compromises because it has insufficient depth. For example, one representation may carry lexical, syntactic, and semantic information simultaneously because no better factorization is available.

When a new block is introduced, all previous parameters should remain trainable so the enlarged network can reorganize:

```text
coarse shallow representation
        ↓
new capacity becomes available
        ↓
new block begins contributing
        ↓
gradients change throughout the stack
        ↓
old and new layers discover a better joint factorization
```

The identity gate is therefore only a mechanism for **continuous model growth at the insertion instant**. It is not intended to preserve the old decomposition permanently.

Embeddings, all active Transformer blocks, MoE/router parameters, final normalization, and LM head should all remain part of the optimization unless an experiment explicitly studies freezing as an ablation.

## 7.5 Possible emergent depth specialization

Progressive depth creates a strongly nonuniform training history across layers.

The first block may see far more raw tokens than the final block. Early layers are therefore under strong pressure to become broadly reusable feature extractors, while later layers are introduced only after simpler predictive structure has already been learned.

A possible emergent hierarchy is:

```text
early blocks
    broad token statistics / lexical structure / common syntax
        ↓
middle blocks
    compositional and semantic relations
        ↓
later blocks
    difficult residual dependencies / abstract prediction
```

This specialization should **not** be assumed or enforced. Ordinary Transformers already exhibit complex co-adaptation, and layer roles are unlikely to divide cleanly into named semantic functions.

However, progressive capacity exposure may produce a different and potentially more structured factorization than training all layers from random initialization on every token. This is an empirical hypothesis worth measuring.

## 7.6 Growth criterion

Do not choose depth transitions only by fixed token counts unless required for the first debugging run.

The intended criterion is a plateau in useful improvement at the current depth. Conceptually measure:

\[
E_D
=
-\frac{\Delta L_{\mathrm{val}}}{\Delta C},
\]

where \(D\) is current depth and \(C\) is wall time or estimated training FLOPs.

Continue training depth \(D\) while the model is obtaining strong loss reduction per unit compute. Add block \(D+1\) when:

```text
minimum exposure at depth D has been reached
AND
validation improvement per compute is below a threshold
FOR
several consecutive measurements
```

A minimum exposure requirement is necessary so noisy validation intervals do not trigger premature growth.

The exact plateau estimator should be simple initially. A fast/slow EMA of validation loss or a linear regression over recent validation checkpoints is sufficient for the first experiment.

The threshold should eventually be calibrated by asking whether continuing at depth \(D\) is less compute-efficient than growing to depth \(D+1\).

## 7.7 Progressive-depth diagnostics

This experiment requires more telemetry than ordinary training.

At minimum record over time:

- current active depth;
- tokens processed at each depth;
- wall time at each depth;
- estimated FLOPs at each depth;
- train and validation loss;
- \(\alpha_{A,l}\) for every active layer;
- \(\alpha_{M,l}\) for every active layer;
- per-layer gradient norm;
- per-layer parameter-update norm;
- residual branch output norm;
- optimizer/LR state at every growth event.

Useful optional diagnostics include:

- activation similarity between adjacent layers;
- how earlier alpha values change after a new layer is inserted;
- how quickly each new gate moves away from zero;
- whether attention and MLP gates activate at different rates;
- whether a newly inserted layer causes a temporary change in gradients of earlier layers;
- validation loss immediately before and after insertion, which should be unchanged within numerical tolerance before any optimization step.

The learned alphas themselves become a measure of whether extra capacity is being used. A layer whose residual gates remain near zero for a long period may indicate that additional depth was introduced too early or is not useful for the current training regime.

## 7.8 Required baseline experiment

The first study should use a smaller model where several complete runs are affordable.

For example, compare an 8-layer 50–100M-class model under a fixed compute budget:

```text
A. conventional fixed 8-layer training
B. progressive 1 → 2 → 3 → ... → 8 training
```

Everything else should remain as identical as possible:

- tokenizer;
- dataset and sampling distribution;
- optimizer family;
- effective batch size where practical;
- precision;
- model width and final architecture;
- evaluation suite.

Compare:

\[
L_{\mathrm{val}}(\text{tokens}),
\]

\[
L_{\mathrm{val}}(\text{wall time}),
\]

and most importantly:

\[
\boxed{L_{\mathrm{val}}(\text{estimated FLOPs})}.
\]

The progressive model will naturally see more raw tokens early because it is cheaper. The experiment is successful only if it improves final quality at matched compute, not merely if it reports a larger nominal token count.

If progressive depth produces a robust improvement, test the same mechanism on the current 12-layer architecture before combining it with token-selection methods.

---

## 7.9 Stage 2: online token learning dynamics

Only after progressive depth is validated should the training system add adaptive token selection.

The goal is to stop treating every target-token occurrence as equally useful throughout the entire run.

The system should estimate online whether a token type is:

```text
already mastered
actively being learned
persistently difficult / possibly noisy
regressing or appearing in a novel context
```

while preserving a permanently random component so the selector cannot trap training in a self-confirming local curriculum.

## 7.10 Per-token online statistics

For every vocabulary token \(v\), maintain lightweight statistics such as:

\[
\mu_v^F,
\qquad
\mu_v^S,
\qquad
\sigma_v^2,
\qquad
n_v,
\]

where:

- \(\mu_v^F\) is a fast EMA of observed loss;
- \(\mu_v^S\) is a slow EMA of observed loss;
- \(\sigma_v^2\) is an online/EMA variance estimate;
- \(n_v\) is the observation count.

For a 65,280-token vocabulary this state is tiny compared with the model and optimizer.

The difference

\[
P_v
=
\mu_v^S-\mu_v^F
\]

acts as a simple learning-progress signal:

```text
large positive P_v
    recent loss is below historical loss
    -> token type is actively becoming learnable

P_v ≈ 0 with low mean loss
    -> token type is already mastered

P_v ≈ 0 with high mean loss
    -> token type remains persistently difficult
```

A normalized form can use:

\[
\hat P_v
=
\frac{\mu_v^S-\mu_v^F}
{\sqrt{\sigma_v^2+\epsilon}}.
\]

## 7.11 Contextual surprise from standardized occurrence loss

Per-token averages alone are insufficient because the same token may appear in very different contexts.

For an observed occurrence \(i\) of vocabulary token \(v\), define:

\[
Z_i
=
\frac{L_i-\mu_v^F}
{\sqrt{\sigma_v^2+\epsilon}}.
\]

This asks:

> **Is this occurrence unexpectedly difficult relative to how the model normally predicts this token?**

This is especially interesting for normally easy tokens.

A token such as a common article may have very low average loss and low variance. If one occurrence suddenly has a very large positive \(Z_i\), the vocabulary item itself is not interesting, but the **context producing that unexpected error may be highly informative**.

Conversely, a token that always has very high loss but little change over time may be noise, an arbitrary identifier, or an intrinsically difficult target. High raw loss by itself should not automatically receive high priority.

Thus the selector should distinguish:

```text
low mean + low surprise
    mastered / low priority

high mean + clear learning progress
    unresolved but learnable / high priority

low mean + unusually high standardized surprise
    novel or difficult context / high priority

high mean + little learning progress + ordinary surprise
    persistently hard / lower priority unless exploration says otherwise
```

Do not reduce this to simply selecting high-variance tokens. Variance is useful primarily as a normalization scale that makes occurrence-level surprise meaningful.

## 7.12 Selected + random training is mandatory

Adaptive selection must retain an explicit random-exploration component throughout training.

Conceptually:

\[
S
=
S_{\mathrm{priority}}
\cup
S_{\mathrm{random}}.
\]

Initial experiments could begin conservatively, for example:

```text
50% priority-selected positions
50% random positions
```

and later explore more aggressive ratios such as:

```text
80% priority-selected positions
20% random positions
```

The exact values are experimental. A nonzero random fraction should remain even late in training.

Random exploration serves several purposes:

- prevents permanently excluding a class the selector currently misunderstands;
- keeps online statistics refreshed;
- detects distribution shifts and novel contexts;
- supplies an approximately unbiased probe of the ignored token population;
- reduces the risk of a self-reinforcing local curriculum.

## 7.13 Frozen early snapshot as an optional auxiliary signal

An inexpensive optional extension is to save a snapshot once the early model has learned common/simple language structure but remains far from convergence.

For example, preserve statistics or a checkpoint around an early milestone such as 5k optimizer steps.

A frozen early model can help distinguish tokens that even a weak model predicts easily from tokens requiring more capacity.

However, do not run the frozen model beside the current model on every training step; that would undermine the compute-saving objective.

Potential inexpensive uses include:

- precomputing token-level/reference statistics for a subset of the corpus;
- storing only per-vocabulary-token early mean/variance baselines;
- periodically evaluating a probe set rather than the whole stream.

The primary online selector should remain based on the current model's own learning dynamics unless experiments demonstrate a clear benefit from the frozen reference.

## 7.14 From better gradient allocation to real wall-clock savings

There are two distinct goals:

```text
better learning per evaluated token
```

and

```text
less computation per raw token
```

If the model computes the complete 65k-vocabulary LM head for every position and only masks losses afterward, adaptive selection may improve gradient quality but will not recover all possible wall-clock savings.

A later implementation can use token-history statistics as a cheap **preselection prior** before the expensive output projection:

```text
target token ID
    ↓
lookup fast/slow loss statistics
    ↓
priority estimate
    ↓
priority positions + random probes
    ↓
full LM head / CE only on selected positions
```

Random probe positions are still evaluated with the real LM head so their true losses can refresh the online statistics and reveal unexpected contexts.

This should only be attempted after the simpler loss-selection version is validated, because pre-head selection changes the optimization and measurement process more substantially.

## 7.15 Coupling token dynamics to depth growth

The two research directions may ultimately provide signals to each other.

For example, at a fixed depth the model may evolve toward:

```text
mastered-token fraction increases
actively-learning fraction decreases
persistent structured errors remain
validation improvement per FLOP falls
```

If random probes show that those remaining errors are repeatable rather than pure noise, this may indicate a **capacity bottleneck** rather than insufficient exposure.

Adding a new Transformer block can then test that hypothesis directly. If previously persistent errors become actively learnable after growth, the new depth is supplying useful representational capacity.

The long-term adaptive loop could therefore become:

```text
DATA STREAM
    ↓
online token learning statistics
    ↓
priority selection + random exploration
    ↓
CURRENT ACTIVE DEPTH D
    ↓
train all active parameters
    ↓
monitor validation improvement per compute
    ↓
continue at D  OR  insert random identity-gated block D+1
```

This creates two independent but interacting allocation decisions:

\[
\boxed{\text{Which training examples deserve gradient compute?}}
\]

and

\[
\boxed{\text{How much model depth deserves to process them?}}
\]

## 7.16 Success criterion for this research direction

The objective is **not** to maximize nominal tokens seen.

A token processed through one Transformer block and a token processed through twelve blocks are not equivalent units of compute. Likewise, a token that never receives a vocabulary projection is not equivalent to a fully trained target position.

Evaluate the strategy using:

- validation loss versus wall time;
- validation loss versus estimated FLOPs;
- downstream quality versus wall time/FLOPs;
- tokens processed at each active depth;
- selected versus random token fractions;
- effective LM-head positions evaluated;
- final fixed-depth quality after the model reaches its target architecture.

The research direction is successful if the target model reaches the same or better final quality with materially less local-GPU time and compute.

A robust 2–3× improvement would already be highly valuable. Larger gains are possible in principle because progressive depth and token selection attack different costs, but claims should be based on matched-compute experiments rather than extrapolation.

---

# 8. Model-capacity experiments

The project should resist scaling automatically, but scaling becomes useful once architectural bottlenecks are understood.

Possible progression:

```text
~500M  current architecture research
~1B    next meaningful local capacity target
larger only if memory/training efficiency remains practical
```

Questions to answer before increasing size:

- Is the current model capacity-limited or data-limited?
- Does router quality saturate before LM quality?
- Does a larger FFN/expert pool improve reasoning more than additional depth?
- Does increasing shared trunk capacity help more than increasing stored MoE capacity?
- Can a 1B model retain useful batch size on the available GPU after offload/recomputation?

Parameter count should be treated as one design variable among many.

---

# 9. Training-system development

Architectural research is only useful if experiments can actually be run locally.

## 9.1 GPU memory reduction

Continue looking for reusable buffers and unnecessary activation retention in:

- SwiGLU/MoE backward;
- QKV projection/replay;
- sparse attention gathers;
- router candidate buffers;
- output projection;
- validation.

Prefer local recomputation or buffer reuse over whole-block activation checkpointing when possible, since full checkpoint replay has previously carried a large throughput penalty.

## 9.2 Better optimizer offload

Current full optimizer offload makes ~500M training possible on 16 GB VRAM.

Future options:

- overlap CPU optimizer work with GPU preparation where dependencies permit;
- lower-precision optimizer history experiments;
- chunk scheduling tuned to PCIe transfer behavior;
- selectively keep hot/small state on GPU;
- memory-map very large optimizer state for >1B experiments if necessary.

## 9.3 Kernel work

Potential optimization targets:

- residual/router score computation;
- top-k selection;
- grouped sparse attention;
- split-ring routed-prefix inference;
- in-place/reused SwiGLU backward buffers;
- fused long-context router pooling;
- reduced host launch overhead.

Performance should be evaluated as end-to-end trained tokens/s, not only isolated kernel speed.

## 9.4 Multi-GPU as an optional acceleration path

The project is local-first, not single-GPU-only by principle.

Simple data parallelism could eventually make short rented multi-GPU experiments useful without changing the model architecture. However, multi-GPU support should not become a requirement for normal development.

---

# 10. Inference development

## 10.1 Better refresh policy

Current routed-prefix inference uses a fixed refresh interval.

Possible alternatives:

- refresh when the current-window query changes sufficiently;
- refresh when router score margins collapse;
- refresh at semantic/document boundaries;
- adaptive interval based on history growth;
- partial prefix replacement instead of complete rebuild.

The important metric is quality versus rebuild cost.

## 10.2 Persistent memory across very long generation

The external store already avoids deep K/V for the full history. Future work could investigate:

- multi-tier memory age buckets;
- compressed old-history representations;
- eviction-time semantic summaries;
- document-level memory indexes;
- explicit memory expiration/relevance decay.

## 10.3 Quantized inference

The training project should eventually produce models that are inexpensive to use locally as well as train locally.

Potential paths:

- BF16/FP16 baseline;
- INT8 weight-only inference;
- 6/5/4-bit weight quantization;
- quantized K/V where appropriate;
- quantized external router store.

Numerical quality and custom-kernel complexity must be measured rather than assumed.

---

# 11. Data and post-training

Architecture alone will not make a small model broadly capable.

Future training stages should eventually explore:

## Continued pretraining

After architecture validation, extend the token budget with a carefully measured high-quality mixture rather than simply maximizing raw token count.

## Instruction tuning

Build a compact, high-quality SFT stage for:

- coding;
- scientific reasoning/writing;
- tool-like structured responses;
- general instruction following.

## Distillation

A small local model can learn behavior from stronger teachers that it would be unlikely to discover from next-token pretraining alone.

Potential targets:

- code reasoning traces;
- scientific explanation;
- structured decomposition;
- debugging;
- long-context retrieval use.

Teacher-generated data should be filtered carefully to avoid merely transferring verbosity or errors.

## Preference/post-training

Only after a competent base/SFT model exists, investigate preference optimization or lightweight reasoning post-training.

---

# 12. Evaluation of capability per resource

Because constrained efficiency is a primary goal, normal benchmark scores are not sufficient.

Track metrics such as:

\[
\text{quality per parameter},
\]

\[
\text{quality per training FLOP},
\]

\[
\text{quality per GiB of VRAM},
\]

\[
\text{long-context recall per active Transformer token},
\]

and

\[
\text{inference quality per unit latency}.
\]

A mechanism that improves a benchmark by 1% while doubling active compute may be less valuable to this project than a mechanism that maintains quality while halving the deep context.

---

# 13. Experimental discipline

For large architectural changes:

1. preserve a known-good reference path;
2. add a small NumPy correctness test;
3. test forward numerical equivalence where applicable;
4. test handwritten backward gradients;
5. perform a short GPU smoke run;
6. measure memory and throughput;
7. evaluate the architectural hypothesis with an ablation;
8. only then fold it into the primary preset.

Avoid changing several conceptual mechanisms in one patch if the result would become impossible to interpret.

---

# 14. Long-term vision

The repository should become a toolkit for asking questions such as:

- How much long-range information does a model truly need in its active context?
- Can semantic context construction replace large amounts of dense attention?
- Can persistent working memory and residual sparse lookup cooperate?
- How should memory be represented when it is too large to pass through the full Transformer?
- How much capability can be recovered through architecture before increasing parameter count?
- What training techniques are most valuable under strict VRAM constraints?
- Can model depth and token-level training compute be allocated adaptively as learning progresses?
- How much of a modern LLM stack can be made understandable and modifiable without giving up practical GPU performance?

The desired outcome is not one frozen model architecture.

It is a **modular local LLM laboratory** in which new model ideas can be implemented, measured, rejected or retained, and iterated on without requiring datacenter-scale resources.

The central constraint remains the central motivation:

> **If scaling is not the immediate answer, we are forced to ask how to make the model better.**
