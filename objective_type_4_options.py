from transformers import T5Tokenizer, T5ForConditionalGeneration
from random import shuffle
import torch

# Load T5 model and tokenizer
model_name = "t5-base"  # Replace with a larger model if needed and resources permit
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate questions
def generate_questions(text, num_return_sequences=1):
    input_text = f"generate questions: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids, max_length=512, num_beams=5, num_return_sequences=num_return_sequences, early_stopping=True)
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

# Function to generate multiple-choice options
def generate_options(answer, distractors, num_options=4):
    # Ensure we have enough distractors
    if len(distractors) < num_options - 1:
        raise ValueError(f"Not enough distractors provided. Need at least {num_options - 1} distractors.")
    
    # Combine answer and distractors, then shuffle
    options = distractors[:num_options - 1] + [answer]
    shuffle(options)
    return options

# Function to create multiple-choice Q&A
def generate_question_and_answer(text, distractors_pool, num_return_sequences=1):
    questions = generate_questions(text, num_return_sequences)
    qa_pairs = []
    for question in questions:
        answer = extract_answer_from_text(question, text)  # Replace this with proper answer extraction logic
        distractors = get_random_distractors(answer, distractors_pool, num_options=4)  # Get 3 random distractors
        options = generate_options(answer, distractors)
        qa_pairs.append({"question": question, "answer": answer, "options": options})
    return qa_pairs

# Function to extract an answer (Dummy Logic, Replace with real implementation)
def extract_answer_from_text(question, text):
    return "Example Answer"  # Placeholder, update with your logic

# Function to get random distractors from a pool
def get_random_distractors(correct_answer, distractors_pool, num_options=4):
    # Exclude the correct answer from distractors
    valid_distractors = [d for d in distractors_pool if d != correct_answer]
    shuffle(valid_distractors)
    return valid_distractors[:num_options - 1]  # Return enough distractors

# Example usage
input_text = "ChatGPT is a language model developed by OpenAI. It is designed to generate human-like text responses."
distractors_pool = [
    "A search engine", 
    "A video streaming platform", 
    "An AI assistant by Google", 
    "A chatbot by Microsoft"
]

# Generate multiple-choice Q&A
qa_data = generate_question_and_answer(input_text, distractors_pool, num_return_sequences=3)

# Display the questions, options, and answers
for i, qa in enumerate(qa_data, 1):
    print(f"Q{i}: {qa['question']}")
    for idx, option in enumerate(qa['options'], 1):
        print(f"   {idx}. {option}")
    print(f"Answer: {qa['answer']}\n")

