# Getting a SPECIFIC real subject into H3 video — without training

_Research, 2026-08-05. Commissioned after the user looked at the best
reference-to-video output of his own car and said: **"it looks legit but it's not
what my car looks like."** He ruled out LoRA training explicitly — training a
model "just for a particular scene" is "just not digestible"._

**THIS FILE IS RESEARCH, NOT MEASUREMENT.** Nothing here has been run on this box
unless a line says so. Claims carry the labels the research produced:

- **VERIFIED** — code or docs found, and the finding was re-checked locally where possible
- **REPORTED** — someone claims it; who is named
- **SPECULATIVE** — inference, not evidence

Measured results live in `RUNBOOK.md` and `bench/profiles.py` in the comfyui
repo. Confirmations get added here later, with dates, as things are actually
tested.

---

## 1. The structural finding — why this problem splits in two

**VERIFIED (re-checked locally):** `grep -rl "arcface|insightface|facexlib|antelope"`
over `comfy/` and `comfy_extras/` returns no real hits — the single match is
`comfy/sd1_tokenizer/vocab.json`, a tokenizer word list, not a model.

H3's ref2va encodes references as **Qwen3-VL vision blocks plus VAE latents**.
There is no face-recognition bottleneck anywhere in the path. That is precisely
**why it works on a car at all**, and it splits the entire research field:

| identity signal | works on a specific **car**? | models |
|---|---|---|
| ArcFace / antelopev2 512-d face vector | ❌ **hard crash, not degradation** — ConsisID raises `RuntimeError("facexlib align face fail")` when no face is present | ConsisID, InstantID, PuLID, IP-Adapter FaceID, ID-Animator, PhotoMaker, EchoVideo, FantasyID |
| generic CLIP/VLM + VAE latent tokens | ✅ | **MiniMax H3 ref2va**, Phantom, VACE, Bernini-R, SkyReels-V3, HunyuanCustom |

**There is no ArcFace for cars and there never will be.** Every "identity
preservation" paper in the face branch is structurally inapplicable to a vehicle.
Worth knowing before reading any of them.

**The consequence for us:** references carry *attributes* — colour, class,
silhouette, trim — not *identity*. That ceiling is architectural. No flag removes
it. Raising it (better crops, more tokens) is possible; removing it is not.

---

## 2. Adapters are a dead end here

**VERIFIED:** no IP-Adapter exists for **any** modern video DiT — not Wan,
Hunyuan, LTX, CogVideoX or H3.

- `ComfyUI_IPAdapter_plus` has been maintenance-only since 2025-04-14. Author,
  verbatim: *"I do not use ComfyUI as my main way to interact with Gen AI anymore."*
- InstantID abandoned 2024-07-18.
- The video DiT families solved image-prompting **natively**, via VAE reference
  tokens — which is why no port was ever written.

**VERIFIED, specifically:** ConsisID's README links to ComfyUI support that does
not exist — it points at `kijai/ComfyUI-CogVideoXWrapper`, which contains zero
"consisid" references. It is also face-only, has no GGUF, and its only 16 GB path
is sequential offload into host RAM we do not have.

---

## 3. What is ALREADY INSTALLED in this tree at ComfyUI 0.30.1

**VERIFIED locally** — these files exist and were read:

| node | file | what it does |
|---|---|---|
| **`BerniniConditioning`** | `nodes_bernini.py` | ByteDance Bernini-R, Apache-2.0. Docstring verbatim: `source_video + ref_video -> ads2v (insert image/video into video)` and `source_video + ref_images -> rv2v (reference-guided video editing)`. **This is the native "put my real car into a generated plate" node.** |
| **Netflix VOID** | `nodes_void.py` | Apache-2.0. RAFT optical-flow warped noise. Removes objects **including their shadows and reflections**. Removal only, no reference input — the correct *erase* step before a composite. |
| **SCAIL-2** | `nodes_scail.py` | Character replacement, SAM3.1-driven, int8 blueprint present. Person only. |
| **`WanPhantomSubjectToVideo`** | `nodes_wan.py` | Phantom — general-subject, Apache-2.0, GGUF available. The RAM-cheapest way to sanity-check the general-subject branch, via the 1.3B variant, without disturbing H3. |
| **`ColorTransfer`** | — | `per_frame` / `uniform` / `target_frame` modes. Video-aware harmonisation for compositing. |

**REPORTED cost for Bernini:** Q4_K_M is 9.66 GB × 2 experts + umT5 ≈ **26 GB
staged** — requires evicting H3 from RAM first. Not a chained test; a separate
session.

**VERIFIED locally:** `WanVaceToVideo` accepts only **one** reference image —
`reference_image[:1]` at `nodes_wan.py:324` silently discards the rest. H3's nine
is a genuine advantage over VACE.

---

## 4. Face swap — viable for the person, with a precise blocker

- **VERIFIED:** ReActor 0.7.0-a2 is alive and registry-active (862k downloads,
  published 2026-05-12). Resolve **`comfyui-reactor`**, NOT `comfyui-reactor-node`
  — the old ID is still marked Active but frozen at 2024-07-03 and points at a
  repo GitHub blocked for TOS violation on 2025-01-16.
- **VERIFIED:** 0.7.0 dropped the `insightface` pip dependency (vendored
  SCRFD/ArcFace) and added **HyperSwap-256** and ReSwapper. FaceFusion 3.8.0's
  default is now `hyperswap_1a_256`, not inswapper.
- **THE BLACKWELL BLOCKER, VERIFIED:** `onnxruntime-gpu ≤ 1.26.0` builds for
  `75;86;89;90-virtual` — **no sm_120**. Only **≥ 1.27.0** (2026-06-18, CUDA 13.0)
  ships native sm_120. **FaceFusion pins 1.26.0 — wrong side of the line. ReActor
  pins nothing and runs `-U`, so it lands correct by default.**
  This box has torch 2.13.0+cu130 and sm_120, with no onnxruntime/insightface/cv2
  installed. **Verify `CUDAExecutionProvider` is actually active** rather than
  silently falling back to CPU.
- **Quality ceiling, REPORTED:** inswapper_128 is a 128 px model; a close-up face
  at 1344×768 is 300–600 px, so it upsamples 3–5×. InsightFace's own benchmark
  scores it **63.3 Realism against 73.7–90.2 for their paid models** — the public
  one is their worst. HyperSwap-256 trades that for different artifacts
  (reported: *"facial features undergo changes, and the background turns into
  colour blocks"*). Face index is recomputed per frame by largest area, so **two
  faces swap places if they cross**. CodeFormer's own README concedes its
  fidelity/quality control is a dial, not a fix.

---

## 5. What is vapour — do not spend time here

**VERIFIED by inspecting the repos:**

- **VideoAnydoor** (SIGGRAPH'25) — a README and one PNG, 5 commits, *"We are now
  organizing the source code"* since Dec 2025.
- **Magic Mirror** (ICCV'25) — 43 image files, **zero `.py` files**.
- **ConceptMaster** — no repo at all.
- "HappyHorse-1.0" — SEO-farm content, not a real project.
- Wan 2.5 / 2.6 / 2.7 — API-only, no weights.

The "insert my specific real object into video" research area is **almost
entirely paper-only**. That is exactly why the already-installed `ads2v` node
matters more than anything on arXiv.

---

## 6. The unsolved problem, stated plainly

**Nothing local relights a filmed car into a generated plate** — directional
light, ground shadows, reflections, camera-motion parallax. `ColorTransfer`
handles global colour statistics only. IC-Light is stills-only.

If we go the composite route, **this is the problem we inherit**, and it is the
one that makes composites look pasted. It is not a detail.

---

## 7. Licensing — flagged, NOT resolved

**REPORTED, needs legal reading before anything ships:**

- No clean commercial path on the swap side: inswapper, SimSwap, GPEN, CodeFormer
  **and** ArcFace/RetinaFace/SCRFD are all non-commercial. HyperSwap is
  ResearchRAIL research-only. FaceFusion's `LICENSE.md` is three lines with no
  license text.
- The `Gourieff/ReActor` HF dataset declares `license: mit` in its metadata —
  **that is wrong for the weights it hosts.**
- H3's own Community License **reportedly** caps commercial use at $20M revenue
  and excludes US/EU/UK/South Korea (TechTimes, MLQ) — **could not be confirmed
  from the Hugging Face model card itself.**

---

## 8. The ladder, cheapest first

1. **Crop references to the subject + `--ref-size max` + a prompt that names the
   identity-bearing detail.** Free. Raises the likeness ceiling; does not remove
   it. *(Being measured 2026-08-05 — see `bench/identity-test.sh`.)*
2. **Real footage as `<Video 1>`**, prompted as an edit instruction.
3. **`BerniniConditioning` `ads2v`** — when a likeness is not enough and the
   *exact* car is required. Native, no training, costs evicting H3 from RAM.
4. **SAM3 + VOID erase + composite** — exactness at the price of the unsolved
   harmonisation problem in §6.
5. **ReActor 0.7.0 + HyperSwap-256** for the face specifically, once
   `onnxruntime-gpu >= 1.27.0` is confirmed working on sm_120.

**LoRA stays last** and is blocked regardless: 27 of 31 GB of host RAM is in use
when the production servers are up.

---

## 9. Follow-ups to confirm by measurement

- [ ] Does crop + `--ref-size max` measurably improve identity? *(running)*
- [ ] Does `ads2v` run at all on this box, and at what RAM cost?
- [ ] Is `CUDAExecutionProvider` actually active with `onnxruntime-gpu >= 1.27.0` on sm_120?
- [ ] Does `WanPhantomSubjectToVideo` (1.3B) fit alongside, as a cheap general-subject check?
- [ ] Confirm H3's licence terms from the model card, not from secondary reporting.
