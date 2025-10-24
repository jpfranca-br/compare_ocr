# Compare OCR Pipeline

This repository extracts license-plate crops from a video source and evaluates multiple OCR backends side-by-side. The typical workflow is:

1. Use a YOLO plate detector (`capture.py`) to watch a video/RTSP stream and save the best crops for each tracked vehicle in `plates/`.
2. Optionally drop a `<image-stem>.txt` file next to any crop to capture your ground-truth transcription.
3. Run `run_all_ocr.py` to execute EasyOCR, PaddleOCR, Tesseract, and OpenAI Vision against the collected images and consolidate the predictions into a CSV for comparison.
4. Inspect results (for example with spreadsheets or the provided helpers) to determine which OCR engine performs best for your use case.

---

## Repository layout

| Path | Purpose |
| ---- | ------- |
| `capture.py` | Streams frames from a video/RTSP source, detects license plates with Ultralytics YOLO, and saves the top crop per tracked ID into `plates/`. |
| `modules/` | Individual wrappers around the supported OCR engines (`easyocr_module.py`, `paddleocr_module.py`, `pytesseract_module.py`, `openai_vision_module.py`). Each wrapper exposes a `run_batch()` helper used by `run_all_ocr.py`. |
| `run_all_ocr.py` | Coordinates the OCR evaluation. It scans a folder for plate images, launches the enabled OCR backends in parallel, and writes a combined CSV containing the predictions for each engine/variant. |
| `make_levenshtein_outputs.py` | Utility for comparing predictions with ground-truth strings using edit-distance metrics. |
| `models/` | Place your YOLO plate-detection weights here (for example `models/license.pt`). |
| `plates/`, `plates_annotated/`, `csv/` | Default output locations for extracted crops, annotated images, and aggregated CSVs. |
| `video/` | Put sample videos here; `capture.py` points to `video/traffic2.mp4` by default. |

---

## Prerequisites

* **Python**: Python 3.10+ is recommended. Create and activate a virtual environment to keep dependencies isolated.
* **System dependencies**:
  * Tesseract OCR (required when using the `pytesseract` backend):
    ```bash
    sudo apt install tesseract-ocr
    ```
  * CUDA drivers/toolkits if you plan to leverage GPU acceleration for PyTorch, PaddleOCR, or YOLO. Follow the vendor instructions for your platform.
* **GPU/accelerator libraries** (optional but recommended for performance):
  * **PyTorch**: install the build that matches your CUDA/cuDNN stack by following the official selector at <https://pytorch.org/get-started/locally/>.
  * **PaddlePaddle**: refer to the official quick-start guide for your OS, CUDA version, and device support at <https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html>.

> **Note**: When you install PyTorch or PaddlePaddle manually to match your hardware, you can remove or adjust the corresponding entries in `requirements.txt` to avoid pip downgrading them.

---

## Installation

1. Clone the repository and enter it:
   ```bash
   git clone <your fork or clone URL>
   cd compare_ocr
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key if you plan to use the `openai_vision` backend:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

5. Place your trained YOLO weights in `models/license.pt` (or update `capture.py` to point to another file). You can train your own plate detector with Ultralytics or reuse an existing model.

---

## Usage

### 1. Extract plates from a video stream

Edit the configuration at the top of `capture.py` if needed:

```python
rtsp_url   = "video/traffic2.mp4"  # local file or RTSP URL
model_path = "models/license.pt"   # YOLO weights
output_dir = "plates"              # where crops will be saved
```

Then run:

```bash
python capture.py
```

The script tracks detections and, after each ID has been visible for `SAVE_DELAY` seconds, writes the best crop (largest bounding box) into `plates/`. Each crop is named after the track ID (e.g., `42.png`).

### 2. (Optional) Add ground truths

For quantitative evaluation, create a text file with the same stem as the image (for example `42.txt` for `42.png`) containing the expected plate string. `run_all_ocr.py` will load it and include the value in the CSV under the `truth` column.

### 3. Run the OCR comparison

Execute the orchestrator with whichever engines you want to benchmark:

```bash
python run_all_ocr.py \
  --in plates \
  --out csv/combined_ocr.csv \
  --enable easyocr paddleocr pytesseract openai_vision \
  --langs_easy pt en \
  --lang_paddle en \
  --lang_tess eng \
  --openai_model gpt-4o-mini
```

Key flags:

* `--in`: folder that contains plate crops (default `plates`).
* `--out`: CSV path (default `csv/combined_ocr.csv`).
* `--enable`: list of OCR engines to run. Supported values: `easyocr`, `paddleocr`, `pytesseract`, `openai_vision`.
* `--langs_easy`, `--lang_paddle`, `--lang_tess`: language packs per backend.
* `--openai_model`, `--openai_ext`, `--openai_q`: configuration for the OpenAI Vision calls.
* `--limit`: optionally cap the number of images processed.
* `--max_workers`: set the parallelism level when running the OCR jobs.

The script writes a CSV where each row represents an image and contains:

| Column | Description |
| ------ | ----------- |
| `image` | Image filename. |
| `truth` | Optional ground-truth string loaded from `<stem>.txt`. |
| `easyocr:*`, `paddleocr:*`, `pytesseract:*`, `openai_vision:*` | Prediction columns (some backends expose multiple variants, e.g., raw text and confidence). |

### 4. Analyze results

Open the CSV in your preferred tool (spreadsheet, pandas, etc.) or run helper scripts such as `make_levenshtein_outputs.py` to compute edit-distance statistics across OCR engines.

```bash
python make_levenshtein_outputs.py --csv csv/combined_ocr.csv
```

This produces additional artifacts for comparing performance.

---

## Tips & troubleshooting

* **GPU availability**: Both YOLO and OCR models benefit from GPUs. Confirm that `torch.cuda.is_available()` returns `True` before running `capture.py` if you expect GPU acceleration.
* **Language packs**: Ensure the corresponding language packs are installed for each OCR engine. For Tesseract, install extra languages with `sudo apt install tesseract-ocr-<langcode>`.
* **Rate limits**: The OpenAI Vision backend calls the OpenAI API. Monitor your usage and ensure the account has sufficient quota.
* **Customizing saving logic**: Adjust `TOP_N`, `SAVE_DELAY`, or `IMG_EXT` in `capture.py` to control how many crops you keep, how long the tracker waits before saving, and the file type used.

---

## Next steps

* Adapt `capture.py` for your specific camera setup or integrate it with a larger video ingestion pipeline.
* Extend `modules/` with additional OCR providers and register them in `run_all_ocr.py`.
* Use the generated CSV to drive automated QA, analytics dashboards, or fine-tune model selection.

