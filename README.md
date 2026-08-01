<div align="center">

# 🎵 Mappernorator

**Turn any song into a playable [Rhythia](https://www.rhythia.com) map.**

Drop in audio, get a `.rhm` / `.sspm` you can drag straight into the game.

![Status](https://img.shields.io/badge/status-work_in_progress-f5a623?style=flat-square)
![Source](https://img.shields.io/badge/source-private-6e7681?style=flat-square)
![Built with PyTorch](https://img.shields.io/badge/built_with-PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![For Rhythia](https://img.shields.io/badge/for-Rhythia-8a2be2?style=flat-square)

</div>

---

> [!IMPORTANT]
> **Work in progress - a personal research project.** The source code is not public and will
> never be released, and it is not intended to be used to gain any advantage in Rhythia.

## ✨ What it does

A small Whisper-style encoder-decoder transformer (~34M parameters) reads the song's mel
spectrogram and emits a stream of note events (`time`, `x`, `y` on the 3x3 grid). It is
trained on maps from rhythia.com, runs on a single 12 GB GPU (an RTX 3060), and ships with a
desktop GUI.

| Mode | What it does |
|:--|:--|
| **Generate** | Pick a song, set difficulty / creativity, choose skillsets, export `.rhm` / `.sspm`. Difficulty spreads, seeds, manual BPM, custom cover, folder batch. **Find ranked** borrows a song's hand-verified timing from osu!'s ranked catalogue, so tempo-changing songs don't rely on BPM auto-detection. |
| **Preview** | Play the song and watch the notes fly in on a 3x3 grid; opens any `.rhm` / `.sspm` too. |
| **Training** | A live loss chart and stats while a model trains. |

## 🧠 How it works

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

## 🎛️ Steerable generation

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

## 📊 Evaluation

Generated maps for held-out human songs are scored on onset **F1** (within 50 ms), grid-cell
accuracy, note-density ratio, and **pattern-mix cosine** (does the map have a human-like blend
of skillsets).

| model | onset F1 | density vs human | pattern-mix |
|:--|:--:|:--:|:--:|
| v1 (early) | 0.54 | 1.41x | - |
| v6 (shipping) | 0.70 | 1.17x | 0.75 |
| v7 (scaled ~3x) | **0.75** | 1.11x | **0.80** |

> [!NOTE]
> v6's F1 is on the stricter frozen-test set; on a common held-out val set v6 and v7 land
> about even. Note placement remains the hardest metric.

## 🔬 The transition-token experiment - a negative result worth writing down

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

The attention fix is the stronger of the two. RMS-normalising the query and key vectors bounds
the score by `sqrt(head_dim)` no matter how large the projections grow, and in training it
holds exactly:

| | cross-attention logits | best validation loss |
|:--|:--:|:--:|
| original run | up to 1,245,475 | 2.1448, then degrading |
| corpus control | up to 88,674 | 2.5007, then erratic |
| **with normalised attention** | **7.5, flat** | **2.0842, still improving at 25k** |

Twenty-five thousand steps, zero rollbacks, zero guard trips, three skipped micro-batches, and
gradient norms *falling* rather than climbing - while still training on the un-fixed corpus, so
the result is conservative.

> [!NOTE]
> Three lessons worth keeping. **A mechanism that explains every measurement is not the same as
> the cause** - the first diagnosis fit every number and was still incomplete; only controls
> separated them. **This class of failure is invisible in the metrics people watch** - loss and
> validation looked healthy while attention scores grew five orders of magnitude, so six
> gradient-side guards fought symptoms for days. And **a placeholder that is also a legal value
> is a bug waiting to happen** - `0.0` meaning "unrated" silently became "easiest" the moment
> something conditioned on it.

> [!WARNING]
> One thing this does **not** yet show: whether the transition token improves map quality. The
> teacher-forced placement metric cannot answer it - that metric feeds the model the true prefix,
> which for this grammar includes the transition token itself, handing it most of the answer.
> Measured: knowing the token drops Y's entropy from 2.509 bits to 1.049. Any cross-grammar
> comparison on that metric is unfair by construction, so the open question is being settled by
> end-to-end generation instead.

## 🗺️ Roadmap

- [x] **RoPE positional embeddings** (custom decoder) - built and trained in the v8 experiment
- [ ] **KV-cache decoding** for faster generation - the RoPE decoder is the groundwork
- [x] **Coordinate-refinement head** - a regression sub-bin offset for beyond-grid precision
- [x] **Rest/gap handling** - an inference-time onset gate so quiet sections stay empty
- [x] **Explicit transition modelling** - built; trains stably now, quality benefit unproven
- [x] **Normalised attention scores** - fixes the failure above; 25k steps clean, best loss yet
- [ ] **Corpus label audit as a gate** - the placeholder-rating bug should have been caught by a
      check, not by a week of divergences

> [!NOTE]
> Three experiments failed to beat the v7 recipe, each differently: **v8** (architecture changes)
> reached its best validation loss at 63k steps and never improved; **v9** (refreshed, larger
> corpus) peaked at 115k and then overfit; **v10** (explicit transitions) destabilised for the two
> reasons above. Both of those are now fixed and the run is stable, so the open question is back
> to the original one - whether any of it produces *better maps* than v7, which only end-to-end
> generation can answer. On this corpus the ceiling has consistently looked like a data limit
> rather than an architecture one.

## 🙏 Credits

Rhythia by CAPO Games. Format details cross-checked against
[rhmParse](https://github.com/yo-ru/rhmParse) and
[pysspm-rhythia](https://github.com/David-Jed/pysspm-rhythia). Concept inspired by OliBomby's
[Mapperatorinator](https://github.com/OliBomby/Mapperatorinator).
