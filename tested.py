import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image

# Step 1: Extract text from file
def extract_text(file):
    text = ""
    if file.name.endswith('.pdf'):
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    elif file.name.endswith('.docx'):
        doc = Document(file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    elif file.name.endswith(('.png', '.jpg', '.jpeg')):
        text = pytesseract.image_to_string(Image.open(file))
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or image.")
    return text

# Step 2: Split text into chunks
def split_text_into_chunks(text, max_length=400):
    words = text.split()
    chunks = [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]
    return chunks

# Step 3: Generate question-answer pairs
def generate_qa_pairs(chunks, model, tokenizer, num_questions=5):
    qa_pairs = []
    for chunk in chunks:
        input_text = f"generate questions and answers: {chunk}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            input_ids, 
            max_length=512, 
            num_beams=4, 
            num_return_sequences=num_questions, 
            early_stopping=True
        )
        for output in outputs:
            qa_text = tokenizer.decode(output, skip_special_tokens=True)
            if "?" in qa_text:  # Split question and answer
                question, answer = qa_text.split("?", 1)
                qa_pairs.append((question.strip() + "?", answer.strip()))
    return qa_pairs

# Step 4: Unified processing function
def process_document(file):
    model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    text = extract_text(file)
    chunks = split_text_into_chunks(text)
    qa_pairs = generate_qa_pairs(chunks, model, tokenizer, num_questions=3)
    return qa_pairs

# Gradio Interface
def display_questions(file):
    qa_pairs = process_document(file)
    output = ""
    for i, (question, answer) in enumerate(qa_pairs, start=1):
        output += f"**Question {i}:** {question}\n\n**Answer:** {answer}\n\n"
    return output

interface = gr.Interface(
    fn=display_questions,
    inputs=gr.File(label="Upload a Document"),
    outputs=gr.Markdown(label="Generated Questions and Answers"),
    title="Question and Answer Generator",
    description="Upload a document (PDF, DOCX, or image) to generate questions with answers displayed below each question.",
)
interface.launch()
