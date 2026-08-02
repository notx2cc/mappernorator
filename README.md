<div align="center">

<br>

# Mappernorator

### Turn any song into a playable [Rhythia](https://www.rhythia.com) map

Drop in audio &nbsp;·&nbsp; get a `.rhm` / `.sspm` you can drag straight into the game

<br>

![Status](https://img.shields.io/badge/status-work_in_progress-E8A33D?style=for-the-badge&labelColor=1c1c22)
![Source](https://img.shields.io/badge/source-private-6E7681?style=for-the-badge&labelColor=1c1c22)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=1c1c22)
![Model](https://img.shields.io/badge/34.3M_params-8A2BE2?style=for-the-badge&labelColor=1c1c22)

<br>

**Jump to**

[![What it does](https://img.shields.io/badge/What_it_does-1c1c22?style=for-the-badge)](#what-it-does)
[![How it works](https://img.shields.io/badge/How_it_works-1c1c22?style=for-the-badge)](#how-it-works)
[![Steering](https://img.shields.io/badge/Steering-1c1c22?style=for-the-badge)](#steerable-generation)
[![Evaluation](https://img.shields.io/badge/Evaluation-1c1c22?style=for-the-badge)](#evaluation)

[![Held-out results](https://img.shields.io/badge/Held--out_results-8A2BE2?style=for-the-badge)](#what-held-out-data-changed)
[![Experiments](https://img.shields.io/badge/Experiments-1c1c22?style=for-the-badge)](#the-transition-token-experiment---a-negative-result-worth-writing-down)
[![Measurement](https://img.shields.io/badge/Measurement-1c1c22?style=for-the-badge)](#measure-what-no-change-at-all-looks-like-first)
[![Roadmap](https://img.shields.io/badge/Roadmap-1c1c22?style=for-the-badge)](#roadmap)

<br>

</div>

---

> [!IMPORTANT]
> **Work in progress - a personal research project.** The source code is not public and will
> never be released, and it is not intended to be used to gain any advantage in Rhythia.

## What it does

A small Whisper-style encoder-decoder transformer (~34M parameters) reads the song's mel
spectrogram and emits a stream of note events (`time`, `x`, `y` on the 3x3 grid). It is
trained on maps from rhythia.com, runs on a single 12 GB GPU (an RTX 3060), and ships with a
desktop GUI.

| Mode | What it does |
|:--|:--|
| **Generate** | Pick a song, set difficulty / creativity, choose skillsets, export `.rhm` / `.sspm`. Difficulty spreads, seeds, manual BPM, custom cover, folder batch. **Find ranked** borrows a song's hand-verified timing from osu!'s ranked catalogue, so tempo-changing songs don't rely on BPM auto-detection. |
| **Preview** | Play the song and watch the notes fly in on a 3x3 grid; opens any `.rhm` / `.sspm` too. |
| **Training** | A live loss chart and stats while a model trains. |

## How it works

**Data.** Readers/writers for `.rhm` (a zip of `map` JSON + audio + cover) and `.sspm` (the
SS+m v2 binary), with byte-verified round-trips and quantum-note support. A rhythia.com API
client fetches ranked maps; audio is decoded to a log-mel spectrogram (80 mels, 10 ms/frame)
with an added onset-strength channel, and maps are preprocessed to compact arrays (mel, notes,
star rating, ranked flag, pattern intensities).

**Model.** An encoder-decoder transformer (~34M parameters). The convolutional frontend is kept
in fp32 to dodge a cuDNN fp16 Conv1d NaN bug on Turing GPUs. Notes map to tokens - `TIME`
(10 ms) plus quantized `X` / `Y` and star-rating conditioning. Generation is sliding-window
(half overlap, with the previous window's notes pre-filled) under grammar-constrained sampling,
so the output is always a valid, ordered note stream.

**Post-processing.** BPM detection (onset autocorrelation plus a fine comb search), BPM-aware
multi-snap quantization (1/1 to 1/16, with the fine grids gated by tempo so 1/16 only applies
to slow songs), duplicate removal, and a travel-speed playability filter calibrated from real
ranked maps.

**Pattern detection.** A geometric detector for 11 of Rhythia's 16 skillsets (Jumps, Slides,
Streams, Stacks, Spins, Bursts, Vibros, Walls, Offgrid, Quantum, Stamina), threshold-calibrated
against human labels. It feeds the conditioning model and a pattern-mix quality metric.

## Steerable generation

Trained on ranked and legacy maps, the conditioning model makes generation steerable. Each
window is prefixed with a control prompt the decoder always sees:

```text
[SOS] [RANKED] [STARS] [DENSITY] [11 skillset toggles] (TIME X Y)* [EOS]
```

Every control can be left as "any" (the model chooses freely), so you can pin only the knobs
you care about - for example *ranked-style, 6-star, heavy streams, no spins*. Two algorithmic
upgrades over the base model:

- **Coordinate-aware loss** - the X/Y bins are not independent classes; near-misses get partial
  credit via a Gaussian over neighbouring bins, targeting a position-accuracy plateau and
  smoother spatial flow.
- **Onset input channel** - the audio's onset-strength envelope is appended as an extra encoder
  channel, an explicit timing prior.

## Evaluation

Generated maps for held-out human songs are scored on onset **F1** (within 50 ms), grid-cell
accuracy, note-density ratio, and **pattern-mix cosine** (does the map have a human-like blend
of skillsets).

| model | onset F1 | density vs human | pattern-mix |
|:--|:--:|:--:|:--:|
| v1 (early) | 0.54 | 1.41x | - |
| v6 (shipping) | 0.70 | 1.17x | 0.75 |
| v7 (scaled ~3x) | **0.75** | 1.11x | **0.80** |

> [!WARNING]
> **These numbers cannot separate the models, and the table's ranking does not survive
> held-out data.** Measured later: re-running one frozen checkpoint with only the sampling
> seed changed moves onset F1 by a mean pairwise 0.033 and pattern-mix by 0.052. The
> margin the table asserts between the top two is **0.006**. See below.

## What held-out data changed

Every model above was ranked on the **validation** split. That is also the split each
checkpoint was *selected* on - training saves a checkpoint whenever validation improves.
So the contest was being decided on the data used to pick the contestants.

The project has always kept a frozen test third: 74 maps, never trained on, never used for
checkpoint selection. It had never been used to compare placement. Scoring the same models
on it inverts the result.

| model | best-val step | validation | **held-out test** | drop |
|:--|:--:|:--:|:--:|:--:|
| shortest run *(the "failed" experiment)* | **63k** | 0.6389 | **0.5600** | -0.079 |
| mid-length run *(shipping default)* | 500k | 0.6737 | 0.5397 | -0.134 |
| longest-trained champion | 300k | **0.7015** | 0.5376 | **-0.164** |

Read the last column. **The better a model looked on validation, the more it lost on data it
had never seen** - and the model abandoned as a failure is the one that transfers best.

Paired per-map, which is the honest test because positions inside one map are heavily
correlated (same song, same mapper) so the effective sample size is 74 maps, not the 95,000
scored positions:

| comparison | mean difference | win rate | sign test |
|:--|:--:|:--:|:--:|
| champion vs shipping default | -0.0015 (0.4 SE) | 35/74 | p = 0.73 - **tied** |
| champion vs the failed run | -0.0230 (4.6 SE) | 17/74 | **p < 0.001** |
| shipping default vs the failed run | -0.0215 (3.7 SE) | 23/74 | **p = 0.002** |

The win rate needs no variance assumption at all - under "no difference" it is a coin flip,
and 17 of 74 is not a coin flip. The two models this project spent three generations ranking
against each other are **statistically indistinguishable**, and the one it discarded beats
both.

The models are not failing in absolute terms. On the same held-out data a marginal predictor
scores 0.115 and a 2nd-order Markov chain given the true previous positions scores 0.262, so
all three sit at roughly **twice the Markov baseline**. They work. They are simply not
distinguishable from one another, and the differences that looked real were selection.

### The likeliest cause is not architecture

The tempting story is that the discarded run's architecture won. The simpler one fits every
row: it is the **shortest** run by a factor of five. Every checkpoint save is a
validation-improvement event against a **36-map** split, so the 300k and 500k runs accumulated
hundreds of selection events against a tiny set while the 63k run stopped early because it
stopped improving. That needs no reference to architecture - and the discarded run changed
three things at once, so a single comparison cannot attribute between them anyway.

> [!NOTE]
> This measures **placement under teacher forcing** - the model is given the true note history
> and asked for the next position. It says nothing about free-running generation quality,
> which is measured separately. A fourth model was dropped from the table entirely: its
> training run began before the test split was frozen and resumed from those weights, so it
> had seen the test maps.

The practical consequence is that a long run can spend most of its compute getting better at
the split it is scored on. The run in progress is now being tracked on held-out placement
directly, to find out whether that is happening to it too.

## The transition-token experiment - a negative result worth writing down

One quality defect survived every model above: at 10-15 notes per second the generator jumps a
median 2.05 grid units where humans compress to 1.00. The idea was to teach transitions
explicitly - give every note a fourth token carrying the quantized distance from the previous
note, so the sequence becomes `TIME → SPACING → X → Y`, and expose that token at inference as
a steering knob for exactly this defect.

**It destabilised training at around 22,500 steps, reproducibly, and no amount of gradient
guarding fixed it.** A capped gradient re-seed, a gradient-norm trip, a spike-rate trip, a
learning-rate cut, per-micro-batch gradient bounding and a starvation hatch all failed - and
one of them made things slightly worse. Training loss sat flat at ~2.2 the whole time and
validation looked healthy, which is precisely why six interventions aimed at gradients never
touched the real problem.

Tracking it down, one measurement at a time:

1. The **per-sample** gradient distribution grew a heavy tail - the 99th percentile rose 493%
   over 3,000 steps on identical samples, while the median moved 4%.
2. The decoder's **cross-attention** logits reached 490,194 and then 1,245,475 - where two
   earlier models on the same architecture, trained far longer, sit between 15 and 173.
3. Those extremes lived almost entirely on **Y** positions (100% of the worst logits in the
   first two decoder layers).
4. And Y is nearly redundant under this grammar. Knowing the previous position, the distance
   `d` and `X`, the circle constraint `(x-px)² + (y-py)² = d²` leaves at most two candidates.
   Measured across 89,638 real note pairs:

| what the model knows when predicting Y | entropy | effective choices |
|:--|:--:|:--:|
| previous note only | 2.979 bits | 7.9 |
| previous note + X | 2.509 bits | 5.7 |
| previous note + transition + X | **1.049 bits** | **2.1** |

That suggested a tidy story: the decoder was running a full cross-attention pass over the
audio at every Y position to decide what amounts to a sign bit, and with nothing useful left
to attend to the attention degenerated.

### The control run refuted it

The story was satisfying, fit every measurement, and was **wrong**. Running the same recipe
with the transition token simply removed - changing nothing else - the model turned at
**~17,000 steps**, about 6,000 steps *earlier* than the run that had the token, with the same
signature: validation improving smoothly and then going erratic, training loss flat throughout,
and the same Y-position cross-attention logits climbing past 88,000.

If the degeneration happens without the token, the token cannot be the cause. Lining the runs
up makes the real correlation obvious, and it is not the grammar:

| run | transition token | community-pack maps | dropout | outcome |
|:--|:--:|:--:|:--:|:--|
| earlier baseline | no | **no** | 0.10 | clean to 115k steps |
| transition-token run | yes | yes | 0.15 | turned ~22.5k |
| control run | no | yes | 0.15 | turned ~17k |

### There were two causes, not one

Five controlled runs, each changing one thing, resolved it. Read down the columns:

| run | transition token | corpus labels | normalised attention | outcome |
|:--|:--:|:--:|:--:|:--|
| original | before X/Y | broken | no | died 22.5k |
| corpus control | **none** | broken | no | turned 17k |
| grammar control | before X/Y | **fixed** | no | **died 6.3k** |
| reordered | **after X/Y** | fixed | no | clean past 10k |
| labels fixed | none | **fixed** | no | **clean to 21k** |
| **normalised** | before X/Y | broken | **yes** | **clean to 25k, best loss** |

Neither column alone predicts the outcome, because there were **two independent failures**
happening at once, and each masked the other:

**1. The grammar.** Placing the transition token before the coordinates leaves Y with about
two possible values, so the decoder runs a full cross-attention pass to decide a sign bit and
the attention degenerates. Fixed either by normalising the attention scores or by moving the
token after the coordinates - the run that kept the old order on a clean corpus died at 6,300
steps, the earliest failure of the whole series.

**2. The corpus labels.** Imported community maps carried `starRating: 0.0` as a placeholder
meaning "unrated" - but 0 is a real bucket, the *easiest* one, and those maps are the hardest
material in the set: 10.7 notes per second against the ranked corpus's 6.8, and 1,788 notes
against 738. So 29% of the training data was telling the difficulty control that zero stars
means maximum density. Re-rating them from note geometry (median 5.02 stars) fixed a run that
otherwise turned at 17,000 steps, changing nothing else.

The attention fix is the stronger of the two. RMS-normalising the query and key vectors strips
the magnitude out of the score no matter how large the projections grow, leaving a scale of
about `sqrt(head_dim)`:

| | cross-attention logits | best validation loss |
|:--|:--:|:--:|
| original run | up to 1,245,475 | 2.1448, then degrading |
| corpus control | up to 88,674 | 2.5007, then erratic |
| **with normalised attention** | **7.5, flat** | **2.0842, still improving at 25k** |

Twenty-five thousand steps, zero rollbacks, zero guard trips, three skipped micro-batches, and
gradient norms *falling* rather than climbing - while still training on the un-fixed corpus, so
the result is conservative.

### The bound was not a bound

`sqrt(head_dim)` is where those logits sit, but calling it a ceiling was wrong, and the full
run caught it. Each normalised vector is multiplied by a small **learnable gain**, initialised
at 1.0 - so the real limit is `sqrt(head_dim) x q_gain x k_gain`, and the gains are free to
grow. Nothing pulled back on them, either: the optimizer's decoupled weight decay is applied
only to matrices, and these gains are 1-D vectors, so they landed in the undecayed group by
construction.

They drifted. Over 50,000 steps the logits climbed 7.2 → 10.1 and the share of them sitting on
Y positions went back from 6% to 84% - the same concentration that preceded the original
blow-up. Reading the weights directly showed the drift was not diffuse but a single layer:

| | mean gain | peak gain | that layer's ceiling |
|:--|:--:|:--:|:--:|
| all 16 gain tensors | 1.0041 | 1.3117 | - |
| decoder layer 1 | - | **1.312** | 8 x 1.720 = **13.8** |
| every other layer | - | 1.06 - 1.14 | 8.9 - 10.3 |

Four orders of magnitude short of the original failure, with gradient norms still falling and
validation still improving - so this was drift, not divergence. But it was drift with no
restoring force, in the exact place the earlier failure began.

The fix is a spring on the gains, and its direction matters: ordinary weight decay pulls a
parameter toward **zero**, which for a multiplicative gain means flattening attention toward
uniform. This pulls toward the **1.0 initialisation** instead, so it resists drift in both
directions and does nothing at all while a gain sits where it started - which, at a mean of
1.0041, is 15 of the 16 tensors. Applied outside the optimizer, so the optimizer's own state
survives a resume untouched.

Watching the peak gain move at last: it had risen monotonically for 15,000 steps, and reversed
within 1,500 of the spring being switched on, while validation kept improving. The spring and
the gradient balance rather than returning it to 1.0 - which is the point. The goal was to
remove an unbounded direction, not to overrule what the model learned.

> [!NOTE]
> The lesson generalises past this project: **a normalisation with a learnable scale is not a
> bound.** It is a bound on the *shape* and a free parameter on the *size*, and the second half
> is easy to forget when the first half is what the technique is famous for. Worth checking
> whether your regularizer actually reaches the parameter you think it does - the split that
> decays matrices but not vectors is a near-universal default, and a learnable gain is a vector.

> [!NOTE]
> Three lessons worth keeping. **A mechanism that explains every measurement is not the same as
> the cause** - the first diagnosis fit every number and was still incomplete; only controls
> separated them. **This class of failure is invisible in the metrics people watch** - loss and
> validation looked healthy while attention scores grew five orders of magnitude, so six
> gradient-side guards fought symptoms for days. And **a placeholder that is also a legal value
> is a bug waiting to happen** - `0.0` meaning "unrated" silently became "easiest" the moment
> something conditioned on it.

> [!WARNING]
> One thing this does **not** show: whether the transition token improves map quality. The
> teacher-forced placement metric cannot answer it - that metric feeds the model the true prefix,
> which for this grammar includes the transition token itself, handing it most of the answer.
> Measured: knowing the token drops Y's entropy from 2.509 bits to 1.049. Any cross-grammar
> comparison on that metric is unfair by construction.

### First look at generated maps

Generating end-to-end sidesteps that problem - the model has to emit its own transition tokens
rather than being handed them. Two models of similar training age, one with the token and one
without, generating for four held-out songs they had never trained on:

| | with token | without token | human | shipping model (20x the training) |
|:--|:--:|:--:|:--:|:--:|
| median aim speed | **10.8 - 23.8** | 20.7 - 33.2 | 16.66 | - |
| 99th-pct aim speed | 1,598 - 2,264 | 2,502 - 2,801 | 70.50 | 33 - 47 |
| centre-cell share | 8.6 - 26.8% | 1.0 - 2.7% | 4.9% | 0.9 - 5.6% |

Two readings, and the first one is a correction:

**The wild aim speeds are immaturity, not the token.** At roughly 5% of a full training run both
models demand ~35x the human 99th-percentile aim speed, and the one *without* the token is worse
on every song. The shipping model needed its full run to settle below the human value. A single
arm would have looked damning; the control is what makes it readable.

**The token does appear to do its job.** Its median aim speed brackets the human figure while the
model without it overshoots on all four songs - which is exactly the defect the token was added
to attack. It overcorrects toward the middle of the grid (a centre-cell share well above human,
where the other model under-uses the centre), and that is the kind of bias the inference-time
steering knob exists to trim.

Both models are far from converged, so this is a direction rather than a verdict.

### The production run

Everything above then went into a single full run on the repaired corpus - normalised
attention, the transition token kept, re-rated community maps, 200 further vetted maps, and
the gain spring added partway through. It is the first run of this line that has stayed
healthy long enough to be compared against the champion rather than against its own failures.

Validation loss against the milestones of the other runs that share its grammar:

| step | this run | normalised arm | transition run |
|:--|:--:|:--:|:--:|
| 20,000 | 2.1190 | - | 2.1448 *(its best, then degrading)* |
| 25,000 | **2.0282** | 2.0842 | *dead* |
| 50,000 | **1.7883** | - | - |
| 75,000 | **1.7057** | - | - |

Ahead of both, at every step where they can be compared.

### Why the champion is missing from that table

An earlier version of this section included the champion's column and reported this run as
ahead of it too. **That comparison was invalid, and the error is worth writing down because
it is the same one the transition token had already caused once.**

Validation loss is mean cross-entropy **per token**. This run emits four tokens per note;
the champion emits three. Adding a low-entropy token class lowers a per-token mean
mechanically - and the transition token also makes the tokens *after* it cheaper, which is
the entropy table from earlier in this document: knowing it drops Y from 2.509 bits to
1.049. Rough arithmetic puts that at ~0.25 nats of free advantage. The gap being celebrated
was 0.002. **The confound was two orders of magnitude larger than the effect, and it ran in
the new model's favour.**

There is no step at which this becomes valid. The grammar difference is permanent, so a
four-token model can never be compared to a three-token one on per-token loss - not at
100k, not at 300k. The metric everyone reaches for first is structurally unavailable for
this architecture.

Two things survive the change:

* **The one comparable channel.** The first token of each note group is emitted under both
  grammars, conditioned on the same note history, so its per-token loss *is* comparable.
  Measured over the entire held-out set: champion **1.669**, this run **1.979** - i.e. this
  run is currently *behind* on the only honest like-for-like number. That is unsurprising at a
  quarter of the training, and the comparison that decides anything is the matched one at
  300,000 steps, where the champion's own final checkpoint sits. It is now logged as a
  curve rather than saved for a single endpoint.
* **Free-running generation quality**, which is protocol-free by construction: it compares
  finished maps against *human* maps rather than against another model, so neither the
  grammar nor the training-length difference touches it.

That reframes the measurement work in this project. It was never a detour around the real
milestone - it built the only instruments that can answer the question, because the obvious
one turns out not to apply here at all.

> [!WARNING]
> Those generation figures are **not** comparable to the evaluation table further up. That
> table uses a different protocol, and the difference is worth more than the gap between any
> two models here - the same checkpoint scores 0.445 or 0.568 depending on one flag. Compare a
> run to itself over time, or re-run every model with byte-identical flags. Nothing else.

Roughly a quarter of the way through its schedule, so still a trajectory rather than a result.

### Measure what "no change at all" looks like first

Building the free-running metric turned up something that invalidated a day of readings and
is probably the most portable lesson here.

Generation is **chaotic in the sampling seed**. One flipped token early in an
autoregressive sample diverges everything after it. Holding the weights frozen, the song
fixed, the difficulty fixed, and varying only the seed:

| seed | median aim speed |
|:--|:--:|
| 1234 | 7.68 |
| 1235 | 3.69 |
| 1236 | 6.66 |
| 1237 | 23.72 |

Mean absolute pairwise difference **10.19**, against a human reference of 13.91. A single
generation says almost nothing.

That one measurement retired four separate conclusions drawn the same day - an apparent
difficulty-conditioning inversion, an apparent checkpoint-to-checkpoint instability
(differences of 6.69 and 8.91, both *inside* the noise), an apparent song-triggered failure
mode, and a "the tail is compressed" verdict. Every one was an n=1 or n=2 reading of a
process noisier than the effects being claimed. None of them could have been caught by
being more careful about any individual reading; what was missing was a **null
distribution**, and nobody had measured one.

Three practical consequences, all of which changed the design:

* **Match the estimator to the noise.** The aggregate is a median, so the matched spread is
  a robust one (MAD), not the pooled figure - 2.96 against 9.03, a 3x difference in what
  counts as significant. The four draws are a tight cluster plus one outlier, not a wide
  spread.
* **Do not discard outliers before a median.** A median of three tolerates one outlier by
  construction; drop it first and you have a median of two, which is a mean. An
  outlier-exclusion gate built earlier that day was removed for exactly this reason - it
  degraded the robustness it was meant to protect.
* **Know what the instrument cannot see.** At nine samples the resolvable difference is
  roughly 20-25% of the reference. It detects large regressions. It cannot see gradual
  quality drift, and any gentle-looking slope should be assumed noise until it clears that
  bar.

The same discipline applies to every metric here, not just this one - and the follow-up is
the more useful half. A per-token-class probe measured on a *fixed* checkpoint moved 8.7%
purely with how many batches it sampled (1.60 / 1.51 / 1.52 / 1.65 at 8 / 12 / 16 / 32
batches), and crucially it did **not** settle with more coverage. The full held-out set
gives **1.669** - outside that entire range, so the sampling was biased rather than merely
noisy, and picking a batch size off a settling curve would have baked that bias in.

The fix was to stop sampling. Dropping the per-class gradient attribution - which the
trajectory does not need, and which is what made the probe expensive - left a forward-only
pass cheap enough to run over the *complete* held-out set in 14 seconds. That has no
sampling error at all, which is a better place to arrive than a carefully chosen sample
size with a documented ±6%.

The general form: when a measurement is noisy, the first question is whether it can be made
*exhaustive* rather than better-sampled. Often the expensive part turns out to be something
the question didn't require.

## Roadmap

- [x] **RoPE positional embeddings** (custom decoder) - built and trained in the v8 experiment
- [ ] **KV-cache decoding** for faster generation - the RoPE decoder is the groundwork
- [x] **Coordinate-refinement head** - a regression sub-bin offset for beyond-grid precision
- [x] **Rest/gap handling** - an inference-time onset gate so quiet sections stay empty
- [x] **Explicit transition modelling** - built; trains stably now, early generation looks right
- [x] **Normalised attention scores** - fixes the failure above; 25k steps clean, best loss yet
- [x] **Corpus label audit as a gate** - the screen that catches unplayable maps now runs as part
      of importing, instead of being something someone remembers to do
- [x] **A restoring force on the learnable attention gains** - the half of the normalisation that
      was never actually bounded
- [ ] **A full run of the fixed recipe** - in progress and ahead of every previous run at every
      comparable step; the first one that gets to answer whether any of this beats the old
      champion

> [!NOTE]
> Three experiments were recorded as failing to beat the v7 recipe: **v8** (architecture
> changes) reached its best validation loss at 63k steps and never improved; **v9** (refreshed,
> larger corpus) peaked at 115k and then overfit; **v10** (explicit transitions) destabilised
> for the two reasons above.
>
> **The first of those verdicts did not survive held-out data** - v8 is the best placer of the
> three on the frozen test set, and was retired on a validation number that inverted. See
> [What held-out data changed](#what-held-out-data-changed). All three verdicts were reached
> the same way, so the other two are worth re-checking on the same footing before they are
> treated as settled.
>
> On this corpus the ceiling has consistently looked like a data limit rather than an
> architecture one - and the held-out result sharpens that, because it suggests part of what
> looked like architecture progress was selection against a 36-map split.

## Credits

Rhythia by CAPO Games. Format details cross-checked against
[rhmParse](https://github.com/yo-ru/rhmParse) and
[pysspm-rhythia](https://github.com/David-Jed/pysspm-rhythia). Concept inspired by OliBomby's
[Mapperatorinator](https://github.com/OliBomby/Mapperatorinator).
