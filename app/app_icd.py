# imports
import base64
from pathlib import Path

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import numpy as np
import random
import pandas as pd
import torch
from ml.ds_icd_llm import PatientEncounterDataset as PatientEncounterDatasetICD
from ml.ds_main_diag_llm import PatientEncounterDataset as PatientEncounterDatasetDRG
import icd10
from annotated_text import annotated_text


# Load the data and model for icd prediction
with open("/local/work/merengelke/ship_former/ds_icd_test_samples.json", "rb") as f:
    data = json.load(f)



model_name_or_path = "/local/work/merengelke/ship_former/models/models"
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(
    "LennartKeller/longformer-gottbert-base-8192-aw512"
)

ds = PatientEncounterDatasetICD(
    "/local/work/merengelke/ship_former/ds_icd_test_samples.json", tokenizer
)
ds_train = PatientEncounterDatasetICD(
    "/local/work/merengelke/ship_former/ds_icd_train_samples.json", tokenizer
)
pats = pd.read_feather("/local/work/merengelke/ship_former/data_raw/patient.ftr")

labels = [label for label in ds_train[0]["label_codes"]]
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

if "annotated_results" not in st.session_state:
    st.session_state.annotated_results = None


# Load the data and model for drg prediction
with open(
    "/local/work/merengelke/main_diagnosis_classifier/icd_pred/samples.json", "rb"
) as f:
    drg_samples = json.load(f)



model_name_or_path_drg = "/local/work/merengelke/ship_former/models/models"
model_drg = AutoModelForSequenceClassification.from_pretrained(model_name_or_path_drg)
if "combined_text" not in st.session_state:
    st.session_state.combined_text = None


# Helper functions
def open_img(image_path: Path) -> str:
    with image_path.open("rb") as file:
        image_data = file.read()
        return base64.b64encode(image_data).decode()


def render_png(png_path: Path) -> str:
    return f"data:image/png;base64,{open_img(png_path)}"


def get_icd_description(icd: str) -> str:
    code = icd10.find(str(icd))
    if code:
        predicted_labels = icd + " (" + code.description + ")"
    else:
        predicted_labels = icd + " ()"  # If no description is found, leave it empty
    return predicted_labels


def write_string_to_file(filename, content):
    try:
        with open(filename, "w") as file:
            file.write(content)
        print("String successfully written to the file.")
    except Exception as e:
        print("An error occurred:", str(e))


def read_file(filename):
    try:
        with open(filename, "r") as file:
            content = file.read()
            return content
    except Exception as e:
        print("An error occurred:", str(e))
        return None


# Given our sample data, you can adapt this to your actual data source
def compare_labels(predicted_labels, true_labels):
    annotations = []
    true_labels_set = set(true_labels)

    for predicted in predicted_labels:
        if predicted in true_labels:
            annotations.append(
                (predicted, "True")
            )  # Using #8AFA8F for a light green color
            true_labels_set.remove(
                predicted
            )  # To ensure each true label is used only once
        else:
            annotations.append(
                (predicted, "False")
            )  # Using #FA8A8A for a light red color
        annotations.append(" ")

    return annotations


# Helper classes
class ICDClassifier:
    def __init__(self, model_path, id_to_label):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
        self.id_to_label = id_to_label  # Ensure you use this instead of id_to_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text):
        inputs = self.tokenizer(
            text, truncation=True, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()

        return self.id_to_label[predicted_class]  # Use self.id_to_label here
        # return predicted_class  # Use this if you want to return the predicted class index


# Config
st.set_page_config(
    page_title="DRG-Main Diagnosis Prediction",
    page_icon=":male_mage:",
    layout="wide",
    initial_sidebar_state="auto",
)

# Sidebar
pages = {"drg": "Main Diagnosis Prediction"}
page_order = ["drg"]


with st.sidebar:
    st.image(render_png(Path("app/.streamlit/SHIPFormer.png")))
    st.write(
        "SHIP-FORMER, developed by SHIP-AI at the Institute"
        " for Artificial Intelligence in Medicine, is an advanced"
        " tool in the Smart Hospital Information Platform that"
        " leverages robust medical evaluations and recommendations"
        " to forecast outcomes such as in-hospital mortality and"
        " insurance denial probabilities. Integrating seamlessly"
        " across hospital systems, it aids physicians in "
        "decision-making and helps administrators optimize operations."
    )
    page = st.selectbox(
        label="Select a page",
        options=[pages[i] for i in page_order],
    )

    st.write("")
    st.write("")

    st.image(render_png(Path("app/.streamlit/ship-ai.png")))

# Main page

# Title and About
st.markdown(
    """
    <style>
    .title-centered {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# st.markdown('<h2 class="title-centered">SHIP-FORMER</h2>', unsafe_allow_html=True)

sentence = "Smart Hospital Information Platform - Foreseeing Outcomes through Robust Medical Evaluations and Recommendations."
formatted_sentence = " ".join(
    [
        f'<span style="font-size: larger; font-weight: bold;">{word[0]}</span>{word[1:]}'
        if word.lower() != "for"
        else word
        for word in sentence.split()
    ]
)

st.markdown(
    f'<h4 class="title-centered">{formatted_sentence}</h4>', unsafe_allow_html=True
)

# Create two columns: left_column and right_column
left_column, right_column = st.columns(2)

# samples only contain patients with more than 4 samples
x = pd.DataFrame([x['label'] for x in data])
df = pd.DataFrame(data)
df['count'] = df.groupby(['patient_id', 'encounter_id'])['patient_id'].transform('count')
df_filtered = df[df['count'] > 4]
pats = df_filtered["patient_id"].unique()
# take 10 random patients
pats = random.sample(list(pats), 10)

# If random_samples doesn't already exist in session state, create it
if "input_text" not in st.session_state:
    st.session_state.input_text = df_filtered[df_filtered["patient_id"] == pats[0]]["text"]

if "patient_data_filtered" not in st.session_state:
    st.session_state.patient_data_filtered = None

if "selected_date" not in st.session_state:
    st.session_state.selected_date = None

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

# Populate the left column with patient selection and information
with left_column:
    # Allow user to select a patient
    patient_id = st.selectbox("**Choose a patient**", pats)
    st.session_state.selected_patient = patient_id
    if patient_id:
        # get all ds samples for this patient for ds[]
        patient_data = df_filtered[df_filtered["patient_id"] == patient_id]

        # extract date by pat data
        dates = [pat.split('sample_date\t')[1].split('\n')[0] for pat in patient_data['text']]
        # dates = [pd.to_datetime(date) for date in dates]

        # add slider to select date
        st.session_state.selected_date = st.select_slider('Select Date', options=dates)  # create a slider to select date
        print(st.session_state.selected_date)

        # Filter the data by selected date
        st.session_state.patient_data_filtered = patient_data[
            patient_data['text'].str.contains(f"sample_date\t{st.session_state.selected_date}")]

        # Get the selected patient's data
        st.session_state.input_text = st.session_state.patient_data_filtered["text"].iloc[0]
        formatted_text = st.session_state.patient_data_filtered["text"].iloc[0].replace("\n", "  \n")
        formatted_text = formatted_text.replace("\t", "  \t")

        st.text_area(
            label="**Patient Information**", value=f"{formatted_text}", height=600
        )

# Populate the right column with the prediction result
with right_column:
    st.markdown("  ")
    st.markdown("  ")

    if patient_id and st.button("Get ICD-10-GE Prediction"):
        # Run inference
        encoding = tokenizer(st.session_state.input_text, return_tensors="pt")
        outputs = model(**encoding)

        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1

        # Display results
        predicted_labels = [
            id2label[idx] for idx, label in enumerate(predictions) if label == 1.0
        ]

        print(predicted_labels)
        print(st.session_state.patient_data_filtered["label"].values[0])
        annotated_results = compare_labels(
            predicted_labels, st.session_state.patient_data_filtered["label"].values[0]
        )

        st.session_state.annotated_results = annotated_results

        # Combine patient_data['text'] and predicted_labels into a single string
        for index, label in enumerate(predicted_labels):
            predicted_labels[index] = get_icd_description(label)

        predicted_labels_str = "\n ".join(map(str, predicted_labels))
        st.session_state.combined_text = (
            patient_data["text"]
            + "\n\n"
            + "Predicted labels: \n"
            + predicted_labels_str
        )
        st.session_state.prediction_made = True

    if st.session_state.annotated_results:
        st.write("**Predicted Labels**")
        annotated_text(*st.session_state.annotated_results)

    # # Check if combined_text is not None before proceeding
    # if st.session_state.prediction_made:
    #     if st.button("Estimate Main Diagnosis (DRG)"):
    #         if st.session_state.combined_text is None:
    #             st.warning(
    #                 "Please first get the ICD-10-CM Prediction before calculating the final DRG code."
    #             )
    #
    #         id_to_label = {idx: item["label"] for idx, item in enumerate(drg_samples)}
    #         classifier = ICDClassifier(model_name_or_path_drg, id_to_label)
    #         txt = (
    #             st.session_state.combined_text
    #             if st.session_state.combined_text
    #             else patient_data["text"]
    #         )
    #         predicted_icd = classifier.predict(txt)
    #         if predicted_icd:
    #             icd = get_icd_description(predicted_icd)
    #         st.write(f"**Predicted Main Diagnosis**")
    #         st.write(icd)
