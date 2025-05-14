import gradio as gr
import fitz  # PyMuPDF
import re
import spacy
import json
import os
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pymongo import MongoClient
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# MongoDB Setup
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["pdf_chatbot"]
collection = db["results"]

# Load LongT5 summarization model (better for large docs)
summarizer_model = "google/long-t5-tglobal-base"
tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")

# QA Model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ------------------ Functions ------------------

def extract_text(file):
    doc = fitz.open(file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_names(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def extract_subject_marks(text):
    pattern = r"(?i)([A-Za-z ]+)[\s:\-]+(\d{1,3})"
    matches = re.findall(pattern, text)
    subjects = {}
    for subject, marks in matches:
        subject = subject.strip().title()
        if subject.lower() not in ["class", "roll", "school", "board", "year", "total", "percentage"]:
            subjects[subject] = int(marks)
    return subjects

def extract_address(text):
    match = re.search(r"Address[:\s]*(.+)", text)
    return match.group(1).strip() if match else ""

def extract_grades(text):
    pattern = r"([A-Za-z ]+)[\s:\-]+([A-F][+-]?)"
    matches = re.findall(pattern, text)
    grades = {}
    for subject, grade in matches:
        subject = subject.strip().title()
        grades[subject] = grade
    return grades

def answer_question(question, text, chunks):
    subject_marks = extract_subject_marks(text)
    question_lower = question.lower()

    if any(x in question_lower for x in ["all", "each", "subject"]) and "mark" in question_lower:
        return "\n".join([f"{subj}: {marks}" for subj, marks in subject_marks.items()])

    for subject in subject_marks:
        if subject.lower() in question_lower:
            return f"{subject} marks: {subject_marks[subject]}"

    context = " ".join(chunks[:3])  # speed
    try:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"⚠️ QA error: {str(e)}"

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    outputs = model.generate(**inputs, max_length=256, min_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def save_to_json(data, filename="output_logs.json"):
    os.makedirs("outputs", exist_ok=True)
    filepath = os.path.join("outputs", filename)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            old_data = json.load(f)
    else:
        old_data = []

    old_data.append(data)
    with open(filepath, "w") as f:
        json.dump(old_data, f, indent=4)

def save_to_mongodb(data):
    collection.insert_one(data)

def save_to_excel(data, filename="output.xlsx"):
    os.makedirs("outputs", exist_ok=True)
    filepath = os.path.join("outputs", filename)

    df = pd.DataFrame([{
        "Name": data.get("name", ""),
        "Roll Number": data.get("roll_number", ""),
        "Class": data.get("class", ""),
        "School": data.get("school", ""),
        "Board": data.get("board", ""),
        "Year": data.get("year", ""),
        "Address": data.get("address", ""),
        "Total": data.get("total", ""),
        "Percentage": data.get("percentage", ""),
        "Subjects": str(data.get("subject_marks", {})),
        "Grades": str(data.get("grades", {})),
        "Questions": "\n".join(data.get("questions", [])),
        "Answers": "\n".join(data.get("answers", []))
    }])

    if os.path.exists(filepath):
        old_df = pd.read_excel(filepath)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_excel(filepath, index=False)

# ------------------ Main Pipeline ------------------

def process_pdf(file, question_input):
    raw_text = extract_text(file)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)

    summary_input = " ".join(chunks[:8])
    summary = summarize_text(summary_input)

    names = extract_names(cleaned_text)
    subject_marks = extract_subject_marks(cleaned_text)
    grades = extract_grades(cleaned_text)
    address = extract_address(cleaned_text)

    # metadata extraction
    roll = re.search(r"Roll\s*Number[:\s]*([0-9]+)", cleaned_text)
    class_ = re.search(r"Class[:\s]*(\w+)", cleaned_text)
    school = re.search(r"School[:\s]*(.+)", cleaned_text)
    board = re.search(r"Board[:\s]*(\w+)", cleaned_text)
    year = re.search(r"Year[:\s]*(\d{4}-\d{2})", cleaned_text)
    total = re.search(r"Total[:\s]*(\d{2,3})", cleaned_text)
    percentage = re.search(r"Percentage[:\s]*(\d{1,3}\.\d+)", cleaned_text)

    # Questions
    questions = [q.strip() for q in question_input.split(",") if q.strip()]
    answers = [answer_question(q, cleaned_text, chunks) for q in questions]

    structured_data = {
        "timestamp": datetime.now().isoformat(),
        "filename": file.name,
        "summary": summary,
        "name": names[0] if names else "",
        "roll_number": roll.group(1) if roll else "",
        "class": class_.group(1) if class_ else "",
        "school": school.group(1).strip() if school else "",
        "board": board.group(1) if board else "",
        "year": year.group(1) if year else "",
        "address": address,
        "subject_marks": subject_marks,
        "grades": grades,
        "total": int(total.group(1)) if total else "",
        "percentage": float(percentage.group(1)) if percentage else "",
        "questions": questions,
        "answers": answers
    }

    save_to_json(structured_data)
    save_to_mongodb(structured_data)
    save_to_excel(structured_data)

    # UI Output
    qa_output = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
    return f"""📄 SUMMARY:
{summary}

Name: {structured_data['name']}
School: {structured_data['school']}
Roll No: {structured_data['roll_number']}
Subjects: {structured_data['subject_marks']}
Grades: {structured_data['grades']}
Address: {structured_data['address']}
Total: {structured_data['total']}, Percentage: {structured_data['percentage']}

❓ QUESTIONS & ANSWERS:
{qa_output}
"""

# ------------------ Gradio UI ------------------

gr.Interface(
    fn=process_pdf,
    inputs=[
        gr.File(label="📄 Upload PDF (up to 10MB)"),
        gr.Textbox(label="Ask Multiple Questions", placeholder="e.g. What is the percentage?, What are the marks in Maths?")
    ],
    outputs=gr.Textbox(label="📘 Output", lines=30),
    title="📄 Smart PDF Chatbot - LongT5 + Multi-Q + Mongo + Excel",
    description="Upload PDF (up to 10MB), ask multiple questions separated by commas. Extract summary, subjects, grades, and more. Data saved to JSON, MongoDB, and Excel."
).launch()
