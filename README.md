
# Project Name

Generation of objective Question and answers through document

## Installation
from transformers 
import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image

### Prerequisites
- Python 3.x
- Dependencies: 
   - `transformers`
   - `torch`
   - `gradio`:This model use for NLP Ui
   - `pytesseract`:This model use for convert image into text (Q&A)

### Steps to Install

1. Clone the repository:
   ```bash
   https://github.com/sandeepsiripuram77/Genaration_Doc_Q-A.git
