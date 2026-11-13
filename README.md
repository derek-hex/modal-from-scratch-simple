# Cat masks and depth on Modal

A small, readable Python script that sends a short video to a GPU on [Modal](https://modal.com).
There, [SAM 3](https://huggingface.co/facebook/sam3) follows a text prompt (default: `cat`)
through the frames and [Depth Anything 3](https://huggingface.co/depth-anything/DA3-BASE)
estimates depth. You get back a mask and a depth map for every frame.

```text
your computer                              Modal
  video ── confirm (type yes) ──────────►  CPU: download model weights (cached after the first run)
                                           GPU: SAM 3 masks, then Depth Anything 3 depth
  output/ ◄──────────── zip ─────────────
```

The whole pipeline is one file, `cat_mask_depth_simple.py`, meant to be read top to bottom.

## What you need

- Python 3.12
- A Modal account
- A Hugging Face account with **approved access** to
  [facebook/sam3](https://huggingface.co/facebook/sam3) (request it on the model page), plus
  a **read** token

## Setup

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m modal setup
Copy-Item .env.example .env
notepad .env
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m modal setup
cp .env.example .env
nano .env
```

In `.env`, paste your Hugging Face token after `HF_TOKEN=` (no quotes, no spaces) and save
it as UTF-8. `.env` is ignored by git; never commit or share it. `modal setup` keeps your
Modal token in your home folder, not in this project.

Open `cat_mask_depth_simple.py` and set `planCredit` to your Modal plan's monthly included
credit in USD (`0` if you have none). It is only used to show how much credit you have left.

## Run

```text
python cat_mask_depth_simple.py
python cat_mask_depth_simple.py my_video.mp4 output-my-video
```

With no arguments it runs the bundled sample clip into `output/`.

- **Launch with `python`.** `modal run` and `modal deploy` are blocked on purpose: they would
  skip the confirmation screen and the `.env` check.
- The input must be a `.webm`, `.mp4`, `.mov`, `.mkv` or `.avi` file, at most 64 MiB and
  at most 1920 x 1080 pixels per frame (portrait is fine). The first 16 frames are processed.
- The output folder must not exist yet. Name it `output` or `output-something` so git
  ignores it.

Before anything starts on Modal, you see a short screen with the hardware, an estimate
and your credit left. Only typing `yes` launches. The first run builds the remote image
and downloads the weights, so it can take several minutes; waiting for a GPU can also
take a few minutes.

## What you get

| File | What it is |
|---|---|
| `masks/mask.0001.png` ... | Full-resolution black and white masks; white is the tracked object. Frames where it is not found are black. |
| `depth/depth.0001.exr` ... | Float32 depth at the model's smaller working resolution. The units are relative, not metres; use an EXR viewer and adjust the display range. |

## Cost

On 2026-11-11, 16 frames took about 45 seconds on an L40S GPU, about **$0.03** at that
day's rates. That estimate is **not a spending cap**. It leaves out the first image build,
the CPU weight download, storage and any retries by Modal. The model weights stay in a
Modal volume named `hf-cache` for reuse, which can add a small storage charge.

After every run, including a failed or interrupted one, check that nothing is still running:

```text
python -m modal app list
python -m modal container list
```

## Settings

The settings sit at the top of `cat_mask_depth_simple.py`. The ones you are most likely
to change:

| Setting | Default | Meaning |
|---|---|---|
| `prompt` | `'cat'` | What SAM 3 should track |
| `maxFrames` | `16` | How many frames to process (1 to 130); more frames take longer and cost more |
| `planCredit` | `30` | Your plan's monthly credit, for the credit-left line |

## Status

This script has not yet been run end to end on Modal. Expect rough edges until that first
run is confirmed.

## Credits

- [SAM 3](https://github.com/facebookresearch/sam3) by Meta and
  [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) by ByteDance Seed
  are installed from their own repositories and are under their own licenses.
- Sample clip: "Kitten jumping on bamboo mat" by Commons user Fumikas Sagisavas, CC0 1.0,
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Kitten_jumping_on_bamboo_mat.webm).
