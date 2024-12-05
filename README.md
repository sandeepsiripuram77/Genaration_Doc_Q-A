
# Project Name

Generation of objective Question and answers through document

## Installation
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image

### Prerequisites
- Python 3.x
- Dependencies: 
   - `transformers`
   - `torch`
   - `gradio`
   - `pytesseract`

### Steps to Install

1. Clone the repository:
   ```bash
