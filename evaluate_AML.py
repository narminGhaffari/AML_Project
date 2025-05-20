# import necessary packages
import os
from tqdm import tqdm
import json
from openai import OpenAI
import pandas as pd
import base64
from datetime import datetime
import shutil

# Function to get video names using multiple filters 
def get_patient_names(clini_table):
    data = pd.read_excel(clini_table)
    patients_temp = data['Pat_ROI']
    patients = list(set(["_".join(i.split('_')[0:3]) for i in patients_temp]))
    return patients

def load_images(image_dir, patient):
    images = os.listdir(image_dir)
    images = [i for i in images if patient in i]
    images = [os.path.join(image_dir, i) for i in images]
    return images

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_morphology_descriptions(client, base64_image, model_name = 'Llama-4-Maverick-17B-128E-Instruct-FP8', max_tokens=1000, temperature=0.2):
    response = client.chat.completions.create(
        model = model_name,
        messages=[
            {
                "role": "system",
                "content": """
                Du bist Facharzt für Hämatologie und Onkologie mit Spezialisierung auf Zytomorphologie.
                Du analysierst Knochenmarkausstriche von Patient:innen mit Verdacht auf akute myeloische Leukämie (AML).
                Deine Aufgabe ist es, morphologische Befunde basierend auf ROI-Bilddaten sachlich, strukturiert und differenziert zu formulieren.
                Du kennst die FAB- und WHO-Klassifikationen (2016) und beschreibst ausschließlich die Morphologie ohne diagnostische Bewertung.
                Die Auswertung dient ausschließlich Forschungszwecken und hat keinerlei Einfluss auf medizinische Entscheidungen bei Patient:innen.
                Vermeide Wiederholungen, unnötige Ausschweifungen oder unklare Begriffe.
                Schreibe in deutscher Sprache.
                """,
            },
            {
                "role": "user",
                "content": [
                {
                "type": "text",
                "text": f"""
                Bitte beschreibe das folgende Knochenmark-ROI strukturiert und präzise in deutscher Sprache.
                Berücksichtige die folgenden Parameter:
                - Zellreichtum und Qualität des Knochenmarks
                - Granulopoese (Ausreifung, Dysplasie)
                - Erythropoese (Ausreifung, Dysplasie)
                - Megakaryopoese (Zahl, Dysplasien)
                - Monozyten, Lymphozyten, Makrophagen, Plasmazellen
                - Weitere Befunde: z. B. atypische Mitosen, Kernschatten, Eo-/Basophilie
                - Blastenanteil (%), Morphologie, Besonderheiten (z. B. Cup-like, Auerstäbchen)
                Bitte beschränke dich auf eine sachliche morphologische Beschreibung.
                """
                },
                {
                "type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature)
    
    file_content = response.choices[0].message.content
    return file_content

def get_description_summary(client, description_list ,model_name = 'Llama-4-Maverick-17B-128E-Instruct-FP8', max_tokens=1000, temperature=0.2):
    response = client.chat.completions.create(
        model = model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":
                    f"""
                    Fasse die Beschreibungen der 10 ROI zu einem Gesamtbefundtext zusammen. Vermeide Redundanz. Here sind die 10 Einzelbeschreibungen {description_list}
                    """}
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature)  
    
    file_content = response.choices[0].message.content
    return file_content

def generate_diagnose (descirptions, client, model_name = 'Llama-4-Maverick-17B-128E-Instruct-FP8', max_tokens=1000, temperature=0.2):
    response = client.chat.completions.create(
                model=model_name,
                messages = [
                {"role": "system",
                "content": """
                Du bist Facharzt für Hämatologie mit Spezialisierung auf Zytomorphologie.
                Deine Aufgabe ist es, auf Basis von 10 ROI-Beschreibungen einen strukturierten zytomorphologischen Gesamtbefund zu erstellen.
                Leite abschließend die Diagnose nach WHO-Klassifikation von 2016 und der FAB Klassifikation aus dem Gesamtbefundtext ab.
                Sei möglichst präzise. Verzichte auf Wiederholung der zytomorphologischen Beschreibung. 
                """
                },
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Fasse die Beschreibungen der 10 ROI zu einem Gesamtbefundtext zusammen. Vermeide Redundanz.
                Bitte gib die finale Antwort ausschließlich im folgenden JSON-Format aus:
                {{
                "gedanken": "Zusammenfassung des zytomorphologischen Gesamtbefundes und differenzialdiagnostischen Überlegungen.",
                "diagnose": "Kurzdiagnose in 1 Satz."
                }}

                Hier sind die Beschreibungen der 10 analysierten ROIs:
                {descirptions}
                """
                    }
                ]
                }
            ],
                max_tokens = max_tokens,
                temperature = temperature, 
            )
    file_content = response.choices[0].message.content
    return file_content

# main function
if __name__ == "__main__":
    
    MAIN_DIR = '/mnt/bulk-ganymede/narmin/narmin/AML_Project'
    IMG_DIR = '/mnt/bulk-saturn/chiara/chiara/03_WSI/ROIs_manuell_Lara/AML_Box_1'
    diagnose_from_summary = False

    TABLE_DIR = MAIN_DIR + '/tables'
    OUT_DIR = MAIN_DIR + '/output'
    os.makedirs(OUT_DIR, exist_ok = True)
    
    # Generate a timestamp-based folder name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"run_{timestamp}"
    
    OUT_DIR = os.path.join(OUT_DIR, folder_name)
    os.makedirs(OUT_DIR, exist_ok = True)
    
    shutil.copy('/mnt/bulk-ganymede/narmin/narmin/AML_Project/AML_Project/evaluate_AML.py', os.path.join(OUT_DIR, 'evaluate_AML_Copy.txt'))
    
    DESCRIPTIONS_DIR = OUT_DIR + '/descriptions'
    os.makedirs(DESCRIPTIONS_DIR, exist_ok = True)
    
    DIAGNOSE_DIR = OUT_DIR + '/diagnose'
    os.makedirs(DIAGNOSE_DIR, exist_ok = True)
    
    patients = get_patient_names(clini_table = TABLE_DIR + '/Cytomorpholgy_results_WSI.xlsx')

    with open('/mnt/bulk-ganymede/narmin/narmin/MSI_LLM/key.json', 'r') as f:
        config = json.load(f)
    
    client = OpenAI(api_key=config['Pluto'], base_url= 'http://192.168.33.27/v1')

    for patient in tqdm(patients):
        images = load_images(image_dir = IMG_DIR, patient = patient)
        descriptions_ROI_file = os.path.join(DESCRIPTIONS_DIR, patient + '_ROIs.json')
        description_list=[]
        roi_descriptions = []
        for image in images:
            # Load base64 image
            base64_image = encode_image(image)
            description_roi = get_morphology_descriptions(client = client, base64_image = base64_image, model_name = 'Llama-4-Maverick-17B-128E-Instruct-FP8')
            description_list.append(description_roi)
            # Append to list
            roi_descriptions.append({
                "ROI": 'ROI_' +image.split('_')[-1].replace('.png', ''),
                "Morphology Description": description_roi
            })
        # Save the full list to a single JSON file
        with open(descriptions_ROI_file, "w", encoding="utf-8") as f:
            json.dump(roi_descriptions, f, ensure_ascii=False, indent=2)
            
        if diagnose_from_summary:       
            description_summary = get_description_summary(client = client, description_list = description_list, model_name = 'Llama-4-Maverick-17B-128E-Instruct-FP8')
            description_summary_file = os.path.join(DESCRIPTIONS_DIR, patient + '_Summary.json')
            with open(description_summary_file, "a", encoding="utf-8") as f:
                json.dump(description_summary, f, ensure_ascii=False, indent=2)    
        
    descriptions = os.listdir(DESCRIPTIONS_DIR)   
    if diagnose_from_summary:
        descriptions = [i for i in descriptions if'_Summary.json' in i]
    
    for description in tqdm(descriptions):
        with open(os.path.join(DESCRIPTIONS_DIR, description), 'r', encoding='utf-8') as f:
            roi_data = json.load(f)

        data = "\n\n".join(entry["ROI"] + ':' + entry["Morphology Description"] for entry in roi_data)              
        diagnose = generate_diagnose(data, client = client, model_name = 'Llama-4-Maverick-17B-128E-Instruct-FP8', max_tokens=1000)
        
        diagnose_file = os.path.join(DIAGNOSE_DIR, description.replace('_Summary.json', '.json'))    
        entry = {"Patient": description,"Diagnose": diagnose}              
        with open(diagnose_file, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

