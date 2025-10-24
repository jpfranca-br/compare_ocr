# 1) Save the 5 files in the same folder:
#    easyocr_module.py paddleocr_module.py pytesseract_module.py openai_vision_module.py run_all_ocr.py

# 2) Install deps you’ll actually use
pip install easyocr paddleocr opencv-python numpy pytesseract openai

https://pytorch.org/get-started/locally/
https://www.paddlepaddle.org.cn/en/install/quick?docurl=undefined

# 3) Have Tesseract installed if using pytesseract (Ubuntu/Debian):
sudo apt install tesseract-ocr

# 4) Have OpenAI key if using openai_vision:
export OPENAI_API_KEY=sk-...

# 5) Run
python run_all_ocr.py --in out --enable easyocr paddleocr pytesseract openai_vision \
  --langs_easy pt en --lang_paddle en --lang_tess eng --openai_model gpt-4o-mini
