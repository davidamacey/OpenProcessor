# Memory titles without a large model: templates, scene words, a very small text model (run R8)

Decision: pending. Nothing is scored yet; this header was committed first (run R8, 17 November 2026).

Registered 17 November 2026, base commit `46cd9c3f2`. Hardware and load of every measured row are
written beside it in the results sections.

## Question

1. **Scene words.** Which is the lightest way to get scene words for a memory ("beach", "snow",
   "birthday cake") that are true to its pictures: (A) the embeddings the product already computes,
   compared with text phrases; or (B) a 22 MB perception model that describes the picture?
2. **Titles.** Does a text model of at most 400 million parameters write better titles from a fact
   sheet than a filled template does?

The owner asked for something very small: no large local model and no hosted model by default;
nothing leaves the host. Design reference: on-device photo apps fill title templates from a few
small models and find themes by comparing a photo's embedding with short text phrases. The words,
phrases and thresholds of this run are our own.

## Data (public only)

| Set | Where | Count | Use |
|---|---|---|---|
| `coco_val2017` (register row of the same name) | pictures of COCO val2017 | 5,000 | all photos |
| COCO instances val2017 | categories and supercategories | 80 categories, 12 supercategories | memories that share a subject; proof categories |
| COCO captions val2017 | `captions_val2017.json` (extracted from the annotations archive) | 5 per photo | the written truth |

Set P and the owner's library are not used. No picture, weight or private path is committed.

- **Test memories:** 60 memories of 3 photos each, seed `20261117`. The 3 photos of a memory share
  one COCO supercategory; supercategories are dealt round-robin from a seeded shuffle.
- **Tuning set (arm A thresholds):** 1,000 other COCO val photos, same seed; no photo is in both sets
  (a unit test checks it). Thresholds are fitted on the tuning set and scored on the test memories.
- **Fact sheets:** 200, same seed. Date range from real calendar dates (2015 to 2025, one day to
  three weeks), the place from 100 world cities and 30 landmarks held in `fact_sheets.py`, people
  0 to 12 (with up to 3 first names when people are named), an event from a list of 28 (public holidays
  of the US and the UK, "wedding", "concert", "hike", "beach day", ...) or none. The first 60 sheets
  belong to the 60 memories, with their event emptied and the kept scene words added. The other 140
  hold 0 to 2 random scene words.

## Scene vocabulary

`scripts/research/title_model/scene_vocabulary.json`: **61 scene words**, written for this run. Each
has synonyms (found in a sentence as whole words), one or two text phrases ("a photo of ..."), and
the COCO category names that count as proof. All 61 occur in COCO val captions (the least frequent,
`concert`, in 14 captions; 44 words occur in 100 or more), so the stop condition "fewer than 40
words occur" does not hold.

**Truth.** A scene word is true for a photo when the word or a synonym is in one of its five
captions (lower case, punctuation removed, whole-word match: "beachfront" is not "beach"), or one of
its proof categories is among its COCO annotations.

## Arms

**Arm A (embeddings the product computes).** For each photo the image embedding, and for each phrase
the text embedding, with each encoder the product serves: MobileCLIP2-S2 through `POST /embed/image`
and `POST /embed/text` (`src/routers/embed.py`), and PE-Core-L14-336 through the product's Triton
models `pe_image_encoder` and `pe_text_encoder` (the `/embed/*` routes serve MobileCLIP only; PE has
no such route, so its two Triton models are called with the product's own preprocessing
`src/utils/pe_preprocess.py` and the PE tokenizer of the pinned `perception_models`). A word scores
the highest cosine of its phrases. A word's threshold is the lowest value with precision at least
0.90 on the tuning set; a word that never reaches 0.90 is dropped for that encoder. Thresholds are
stored per encoder name.

**Arm B (Mirage).** `cactus.Mirage().describe(<file>)` per photo, CPU, `CACTUS_TELEMETRY=0`,
`DO_NOT_TRACK=1`, `HF_HUB_OFFLINE=1` (the weights and engine are fetched once beforehand, the run
itself cannot reach the network). The description is the text of every string in its answer (scene
setting, lighting, object names and their attributes). A word is found when the word or a synonym is
in it (lower case, punctuation removed, whole-word match). A second variant uses the documented
option `sections=["scene"]` (scene description only); it is reported next to the default one and
changes no bar.

**Keep rule (both arms).** A memory keeps a word that is found for at least 2 of its 3 photos; at
most 3 words, the highest-scoring first (arm A: mean cosine minus threshold; arm B: how many photos).

**Titles, three ways on the same sheets.**
- *The plain template* `"<place>, <month year>"` (no place: `"<month year>"`): the reference and the
  fallback.
- *The template table* `scripts/research/title_model/title_templates.json`: rows per scene word
  family (hiking, beach, snow, dinner, concert, birthday and 11 more, plus "none"), each with the
  variants plain, place, one named person, a group (2 or 3 named people). The row is picked by the
  sheet's event if it has one, else by its first kept scene word; the variant by what the sheet holds.
  A sheet with no event and no scene word gets the plain template.
- *A text model* that gets the fact sheet as text and answers one JSON object.

For every way the subtitle is the place (if any) and the date range, except a text model's own.

### The prompt, the schema and the decode settings (fixed before anything is scored)

System message:

```
You write short titles for photo memories. Use only the facts given. Never add a place, a name, a number, a date or an event that is not listed. Answer with one JSON object and nothing else.
```

User message (`{...}` is filled from the sheet; "none" when the sheet holds nothing):

```
Facts:
Dates: {dates}
Place: {place}
People: {people}{named}
Event: {event}
Scene words: {scene_words}

Write a title of at most 8 words and a subtitle of at most 12 words. Answer as {"title": "...", "subtitle": "..."}.
```

JSON schema (enforced as a grammar by `llama-cpp-python`, `response_format` of type `json_object`
with this schema):

```json
{"type": "object",
 "properties": {"title": {"type": "string", "maxLength": 50},
                "subtitle": {"type": "string", "maxLength": 70}},
 "required": ["title", "subtitle"], "additionalProperties": false}
```

Decode: temperature 0, seed 20261117, at most 48 new tokens, 4 threads, context 1024, one title at a
time, no retry. A title that does not parse counts as invalid JSON.

No picture leaves the host in any arm.

## Candidates (licence lines as the pages stated them on 17 November 2026)

Size limit for every model this run adds: at most 400 million parameters and at most 500 MB on disk
as run.

| Candidate | Arm | Size | Licence as stated | Page |
|---|---|---|---|---|
| MobileCLIP2-S2 (in the product) | A | 35.7M parameters | weights `apple-amlr` (Apple's research licence; the user downloads them) | `ATTRIBUTION.md` |
| PE-Core-L14-336 (in the product) | A | in the product | Apache-2.0 | `ATTRIBUTION.md` |
| Mirage (Cactus Compute) | B | 51.3M parameters (vendor), weights 22 MB on disk | model page: "Apache-2.0"; engine page: "License: apache-2.0", "Apache 2.0, including the prebuilt runtime binaries" (package metadata) | https://huggingface.co/Cactus-Compute/mirage, https://huggingface.co/Cactus-Compute/cactus |
| SmolLM2-360M-Instruct | titles | "0.4B params" | "License: apache-2.0" | https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct |
| SmolLM2-135M-Instruct | titles | "0.1B params" | "License: apache-2.0" | https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct |
| LFM2-350M (Liquid AI) | titles | 354,483,968 parameters | "License: lfm1.0" (LFM Open License v1.0): rights to use the model commercially end if the company's annual revenue exceeds USD 10 million | https://huggingface.co/LiquidAI/LFM2-350M, https://www.liquid.ai/lfm-license |

Looked at and not run:

- **Desert Ant Labs** (https://desertant.com/blog/introducing-desert-ant-labs/): 18 small task
  models (speech, audio clean-up, masking of personal data, language detection, video clips). None
  writes a caption or a title, the page names no Linux server runtime and no licence for the
  weights. Look again if they publish a caption model.
- **Google's 270M instruction model of the third generation** (the 270M model page of Google on Hugging
  Face): its licence is Google's own terms and the repository is gated. Not permissive; not run.
  (Its short name is left out of this note because the repository's naming scan bans it.)
- **Qwen3 0.6B / 1.7B, SmolLM3 3B, Phi-4-mini**: above the owner's size limit; not run.

**Licence rule.** Every candidate is measured and its licence recorded next to its result. The
`Decision:` line may name as the default only a model whose weights and runtime are under a
permissive licence (Apache-2.0, MIT or equivalent). A model under other terms is named only as an
option the user switches on and downloads themselves; its weights are never in an image and never
downloaded by default.

**Runtimes.** Text models on the CPU with `llama-cpp-python` 0.3.37 (llama.cpp, MIT; its vendored
llama.cpp is commit `a956a852c9c8bc1fb07206364838ddef299178c0`) and a `Q8_0` GGUF. The GGUF is
converted by the run from the authors' own `safetensors` with that commit's
`convert_hf_to_gguf.py --outtype q8_0`. The card names the `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF`
file as the authors' own; its README says it was converted by a third party
(`ngxson/SmolLM2-360M-Instruct-Q8_0-GGUF`, "GGUF-my-repo"), so it is not used: "never a third
party's GGUF". Mirage runs with `cactus-compute` 4.2.3 (engine 4.1.1, wheels for Linux x86-64), CPU
only. Packages live in the run's own venv (`venvs/R8` on the scratch drive, outside the repository); exact pins in
`scripts/research/title_model/requirements.txt`.

## Rubric and bars (registered; the bar is never moved)

An arm or a model must meet every line of its part; a missed line closes it.

Scene words (arm A per encoder, and arm B):

| Check | Measured by | Bar |
|---|---|---|
| A kept word is true for its memory (true for at least 2 of the 3 photos) | the script, all 60; then one Haiku agent looks at the 3 photos of 30 memories | at most 3 of 60 memories with a wrong word (script); at most 2 of 30 (Haiku) |
| A memory gets at least one word | the script | at least 70% of the 60 |
| Cost per photo, CPU | arm A: the time of the cosine against all phrases, embeddings given (no model runs); arm B: the script, 4 threads, 3 repetitions, load average beside each row | arm A: at most 1 ms; arm B: 95th percentile at most 500 ms and resident memory at most 500 MB |
| It runs on this host | arm B: `cactus-compute` installs in the run's venv, describes one photo, and sends nothing off the host | yes; otherwise the arm closes with the reason |

Titles:

| Check | Measured by | Bar |
|---|---|---|
| No invented fact (allowed: the sheet's place, numbers, event and scene words, words from its dates: month, year, season, weekday, "weekend", "week"; the list is `fact_sheets.invented_facts`) | the script, all 200 | at most 2 of 200 titles fail |
| One valid JSON object with `title` and `subtitle` (text models) | the script, all 200 | at least 99% without a retry |
| The place is in the title or subtitle when the sheet has one | the script | at least 95% |
| At most 8 words, plain text, no emoji, no quotes | the script | at least 95% |
| Specific and natural for a photo memory, 1 to 5 | one Haiku agent, 50 titles per way, blinded and shuffled | the template table: mean at least the plain template's plus 0.5. A text model: mean at least the template table's plus 0.5 |
| Time per title, CPU (text models; 4 threads, one title at a time) | the script, 3 repetitions, load average beside each row | 95th percentile at most 2 s; resident memory at most 1 GB |

Statistics: every rate carries its sample size and a 95% Wilson interval; a difference of mean
ratings carries a paired bootstrap interval (10,000 resamples). A timing is 3 repetitions with the
load average beside each row; it runs on a quiet host.

Interpretation fixed with the registration (not a moved bar): words counted by the script for "at
most 8 words" are space-separated tokens of the title; a fact sheet also holds up to 3 first names
when people are named, because the template variants "one named person" and "a group" need them;
the invented-fact check is applied to the title and the subtitle.

## If a bar is missed

Scene words missed by both arms: the template table is filled from calendar entries, holidays and
places only, and no scene-words step is built. Text models missed: titles are filled templates, and a
text model stays an option the user turns on. The note says which line each arm missed and by how
much.

## Results

Not yet scored.
