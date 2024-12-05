import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image

# Function to extract text from PDF, DOCX, or image
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

# Function to split text into smaller chunks
def split_text_into_chunks(text, max_length=400):
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence.split()) <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence.split())
        else:
            chunks.append(' '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = len(sentence.split())

    if current_chunk:
        chunks.append(' '.join(current_chunk) + '.')
    return chunks

# Function to generate questions and answers
def generate_qa_pairs(chunks, model, tokenizer, qa_pipeline, num_questions=5):
    qa_pairs = []
    for chunk in chunks:
        input_text = f"generate questions: {chunk}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        try:
            outputs = model.generate(
                input_ids, 
                max_length=512, 
                num_beams=4, 
                num_return_sequences=num_questions, 
                early_stopping=True
            )
            for output in outputs:
                question = tokenizer.decode(output, skip_special_tokens=True).strip()
                print(f"Generated Question: {question}")  # Log generated question
                
                try:
                    # Attempt to answer the question using QA pipeline
                    answer = qa_pipeline(question=question, context=chunk)['answer']
                    print(f"Generated Answer: {answer}")  # Log generated answer
                except Exception as e:
                    answer = f"Error generating answer: {e}"  # Log error if QA fails
                    print(answer)
                
                qa_pairs.append((question, answer))
        except Exception as e:
            print(f"Error during question generation: {e}")
            qa_pairs.append(("Error generating question", f"Error: {e}"))
    return qa_pairs

# Function to process the document and generate questions and answers
def process_document(file):
    model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    text = extract_text(file)
    chunks = split_text_into_chunks(text)
    qa_pairs = generate_qa_pairs(chunks, model, tokenizer, qa_pipeline, num_questions=3)
    return qa_pairs

# Function to display the questions and answers in markdown format
def display_questions(file):
    qa_pairs = process_document(file)
    output = ""
    for i, (question, answer) in enumerate(qa_pairs, start=1):
        output += f"### Question {i}: {question}\n\n**Answer:** {answer if answer else 'No answer generated.'}\n\n"
    return output

# Create Gradio interface
interface = gr.Interface(
    fn=display_questions,
    inputs=gr.File(label="Upload a Document"),
    outputs=gr.Markdown(label="Generated Questions and Answers"),
    title="Question and Answer Generator",
    description="Upload a document (PDF, DOCX, or image) to generate questions with answers displayed below each question.",
)

# Launch the interface
interface.launch()
