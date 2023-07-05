from PyPDF2 import PdfReader
import streamlit as st
from transformers import LongformerTokenizer, LongformerModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import hashlib

# Load the Longformer model and tokenizer
longformer_model_name = 'allenai/longformer-base-4096'
longformer_tokenizer = LongformerTokenizer.from_pretrained(longformer_model_name)
longformer_model = LongformerModel.from_pretrained(longformer_model_name)

# Load the MedBERT model and tokenizer
medbert_model_name = 'dmis-lab/biobert-v1.1'
medbert_tokenizer = AutoTokenizer.from_pretrained(medbert_model_name)
medbert_model = AutoModelForSequenceClassification.from_pretrained(medbert_model_name)

# Define the PICOS criteria and context
picos_criteria = {
    'Population': [],
    'Intervention': [],
    'Comparison': [],
    'Outcome': [],
    'Study_design': []
}
context = "Stick to the PICOS criteria of population, intervention, comparison, outcome, and study design to decide whether the paper should be accepted or not."

# Set password for the app

# Streamlit app
st.title("Research Paper Evaluation")

# Input for PICO criteria
st.subheader("Input PICO criteria")

population = st.text_input("Population: (separate multiple values with comma)").split(',')
intervention = st.text_input("Intervention: (separate multiple values with comma)").split(',')
comparison = st.text_input("Comparison: (separate multiple values with comma)").split(',')
outcome = st.text_input("Outcome: (separate multiple values with comma)").split(',')
study_design = st.text_input("Study Design: (separate multiple values with comma)").split(',')

picos_criteria['Population'] = [value.strip() for value in population]
picos_criteria['Intervention'] = [value.strip() for value in intervention]
picos_criteria['Comparison'] = [value.strip() for value in comparison]
picos_criteria['Outcome'] = [value.strip() for value in outcome]
picos_criteria['Study_design'] = [value.strip() for value in study_design]

# Input for Research Paper PDF
st.subheader("Upload Research Paper PDF")
pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

# Evaluation button
if st.button("Evaluate") and pdf_file is not None:
    # Extract text from PDF
    pdf_reader = PdfReader(pdf_file)
    pdf_text = ''
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    # Chunk the research paper using Longformer
    max_chunk_length = 4096
    chunks = [pdf_text[i:i+max_chunk_length] for i in range(0, len(pdf_text), max_chunk_length)]

    # Initialize variables to store overall results
    overall_accept_probability = 0.0

    # Initialize progress bar
    progress_bar = st.progress(0)

    # Process each chunk separately
    for i, chunk in enumerate(chunks):
        # Encode the chunk using Longformer tokenizer
        encoding = longformer_tokenizer.encode_plus(
            chunk,
            add_special_tokens=True,
            max_length=4096,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Use Longformer for chunk encoding
        with torch.no_grad():
            longformer_output = longformer_model(input_ids, attention_mask=attention_mask)

        # Get the contextual embeddings from Longformer output
        contextual_embeddings = longformer_output.last_hidden_state

        # Combine the PICOS criteria and context
        pico_text = ' '.join([f"{key}: {' '.join(values)}" for key, values in picos_criteria.items()])
        text = f"{context} {chunk} {pico_text}"

        # Encode the combined text using MedBERT tokenizer
        medbert_input = medbert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        medbert_input_ids = medbert_input['input_ids']
        medbert_attention_mask = medbert_input['attention_mask']

        # Use MedBERT for inference
        with torch.no_grad():
            medbert_logits = medbert_model(input_ids=medbert_input_ids, attention_mask=medbert_attention_mask).logits

        # Convert logits to probabilities using softmax
        medbert_probabilities = torch.softmax(medbert_logits, dim=1)
        accept_probability = medbert_probabilities[0][1].item()  # Probability of accepting the chunk

        # Add current chunk's results to overall results
        overall_accept_probability += accept_probability

        # Update progress bar
        progress_text = f"Chunk {i+1}/{len(chunks)}"
        progress_bar.progress((i + 1) / len(chunks))

    # Calculate the average acceptance probability across all chunks
    average_accept_probability = overall_accept_probability / len(chunks)

    # Define the acceptance threshold
    threshold = 0.6

    # Make the accept/reject decision based on the average probability
    if average_accept_probability >= threshold:
        decision = "Accept"
    else:
        decision = "Reject"

    # Display the result
    st.subheader("Result")
    st.write("Decision:", decision)
    st.write("Probability:", average_accept_probability)
