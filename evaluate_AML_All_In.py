# import necessary packages
import os
from tqdm import tqdm
import json
from openai import OpenAI
import pandas as pd
import base64
from datetime import datetime
import shutil
import csv
from pathlib import Path
import torch
import re

# Function to get video names using multiple filters 
def get_patient_names(clini_table):
    if '.xlsx' in  clini_table:
        data = pd.read_excel(clini_table)
        patients_temp = data['Pat_ROI']
        patients = list(set(["_".join(i.split('_')[0:3]) for i in patients_temp]))
    else:
        data = pd.read_csv(clini_table)
        patients = list(set(data['Patients'].tolist())) 
    return patients

def load_images(image_dir, patient):
    images = os.listdir(image_dir)
    images = [i for i in images if patient+ '_' in i]
    images = [os.path.join(image_dir, i) for i in images]
    if not len(images) == 10:
        raise ValueError(f"Patient {patient} has {len(images)} images, expected 10.")
    return images

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_diagnose (encoded_images, client, model_name, max_tokens, temperature):
    
    messages=[
        {
            "role": "system",
            "content": """
            Du bist Facharzt für Hämatologie mit Spezialisierung auf Knochenmarkzytomorphologie. 
            Deine Aufgabe ist es, 10 übergebene ROI-Bilder eines Knochenmarkausstrichs mikroskopisch zu analysieren. 
            Nutze ausschließlich morphologische Kriterien gemäß FAB- und WHO-Klassifikation zur strukturierten Analyse (z. B. Kernform, Nukleolen, Granula, Dysplasiezeichen), jedoch **nicht zur Subtypisierung**.

            Ziel: Erstelle auf Basis der Bilddaten einen **strukturierten zytomorphologischen Gesamtbefund** sowie eine **übergeordnete morphologische Diagnosekategorie**.
            
            Wichtige Einschränkungen:
            - Keine Subtypisierung (z. B. M0–M7)
            - Keine Aussagen zu Zytochemie, Immunphänotyp, Molekulargenetik oder POX-Färbungen
            - Immer den geschätzten **Blastenanteil in Prozent** angeben
            - Die Antwort erfolgt ausschließlich auf **Deutsch**, ohne Wiederholung einzelner ROI-Beschreibungen
            - Gib die Antwort **nur im definierten JSON-Format** aus
            """
        },
        {
            "role": "user",
            "content": [
            {
            "type": "text",
            "text": f"""
            Bitte analysiere die folgenden 10 Knochenmark-ROIs anhand dieser Kriterien:

            - Qualität des Ausstrichs: Zellreichtum, Bröckelchengehalt
            - Granulopoese: Reifung, Dysplasiezeichen
            - Erythropoese: Reifung, Dysplasiezeichen
            - Megakaryopoese: Zellzahl, Dysplasiezeichen
            - Lymphopoese: Morphologie, Reifung
            - Weitere Zellen: Monozyten, Makrophagen, Plasmazellen
            - Besonderheiten: Atypische Mitosen, Kernschatten, eosinophile/basophile Zellen, Auerstäbchen, Cup-like Kerne etc.
            - Blasten-Anteil (%) und morphologische Auffälligkeiten
            
            Gib die finale Antwort ausschließlich im folgenden JSON-Format aus:

            {{
            "gedanken": "Zusammenfassung des zytomorphologischen Gesamtbefundes und diagnostische Überlegungen.",
            "Qualität des Ausstrichs-Markbröckelchen": "Gehalt an Markbröckelchen im Ausstrich", 
            "Qualität des Ausstrichs-Zellgehalt": "Zellreichtum im Knochenmark", 
            "Erythropoese": "Ausreifung und Dysplasiezeichen",
            "Granulopoese": "Ausreifung und Dysplasiezeichen",
            "Megakaryopoese": "Ausreifung und Dysplasiezeichen",
            "Lymphopoese": "Ausreifung und Dysplasiezeichen",
            "Blastengehalt": "Zahl in Prozent %", 
            "Besonderheiten":  "Cup-like, Auerstäbchen, Nukleolen, Makrophagen, z. B. atypische Mitosen, Kernschatten, Eo-/Basophilie",
            "Diagnose": "Kurzdiagnose in 1 Satz",
            }}
            Hier sind die 10 ROI-Bilder:
            """
                },
                *encoded_images  
            ]
        }
    ]
    
    response = client.chat.completions.create(
                model=model_name,
                messages = messages,
                max_tokens = max_tokens,
                temperature = temperature, 
            )
    file_content = response.choices[0].message.content
    return file_content

import re
import json

def fix_json_string(raw_text):
    # Step 1: Cleanup markdown markers and leading/trailing whitespace
    raw_text = raw_text.strip()
    raw_text = re.sub(r'^```json\n?', '', raw_text)
    raw_text = re.sub(r'\n```$', '', raw_text)

    # Step 2: Fix unquoted values
    def replacer(match):
        key = match.group(1)
        value = match.group(2).strip()

        if re.match(r'^-?\d+(\.\d+)?$', value) or value in ['true', 'false', 'null']:
            return f'"{key}": {value}'
        elif value.startswith('"') and value.endswith('"'):
            return f'"{key}": {value}'  # already quoted
        else:
            value = value.replace('"', '\\"')
            return f'"{key}": "{value}"'

    # Handles cases where value is not quoted properly
    pattern = r'"([^"]+)"\s*:\s*([^",\n]+(?:\s[^",\n]+)*)'
    fixed_text = re.sub(pattern, replacer, raw_text)

    # Step 3: Remove trailing commas before closing braces/brackets
    fixed_text = re.sub(r',\s*([}\]])', r'\1', fixed_text)

    # Step 4: Try loading into JSON
    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError as e:
        print("❌ JSON decode error:", e)
        print("🛠 Fixed text preview:\n", fixed_text[:500])
        raise


def write_results_to_csv(input_dir, output_file):
    input_dir = Path(input_dir)
    rows = []
    all_keys = set()

    # Loop through all JSON files
    for file in input_dir.glob("*.json"):
        print(file)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        patient_id = data.get("Patient", file.stem)
        patient_id = patient_id.replace('_ROIs.json', '')
        
        # Step 1: Get the Diagnose or Ergebnis field
        raw_diagnose = data.get("Ergebnis")

        # Step 2: Clean up Markdown formatting from Diagnose
        if raw_diagnose.startswith("```json"):
            raw_diagnose = raw_diagnose.strip("`").strip()
            first_newline = raw_diagnose.find('\n')
            raw_diagnose = raw_diagnose[first_newline+1:]  # Skip first line
            if raw_diagnose.endswith("```"):
                raw_diagnose = raw_diagnose.rsplit("```", 1)[0]

        # Step 3: Parse the cleaned JSON string
        try:
            diagnose_dict = json.loads(raw_diagnose)
        except:
            
            diagnose_dict = fix_json_string(raw_diagnose)
            

        # Add patient as a separate column
        diagnose_dict["Patient"] = patient_id
        all_keys.update(diagnose_dict.keys())
        rows.append(diagnose_dict)

    # Assuming you already have a list of dicts in `rows` and all keys in `fieldnames`
    df = pd.DataFrame(rows)

    df = df[["Patient"] + [col for col in df.columns if col != "Patient"]]
    df.to_excel(output_file, index=False)

        
# main function
if __name__ == "__main__":
    
    MAIN_DIR = '/mnt/bulk-ganymede/narmin/narmin/AML_Project'
    IMG_DIR = '/mnt/bulk-saturn/chiara/chiara/03_WSI/ROIs_manuell_Lara/AML_Box_1'
    
    only_generate_csv = False

    if not only_generate_csv:
        
        TABLE_DIR = MAIN_DIR + '/tables'
        OUT_DIR = MAIN_DIR + '/output'
        
        model_name = 'medgemma-27b-it-q6'
        
        os.makedirs(OUT_DIR, exist_ok = True)
        
        # Generate a timestamp-based folder name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"run_{timestamp}_{model_name}"
        
        OUT_DIR = os.path.join(OUT_DIR, folder_name)
        os.makedirs(OUT_DIR, exist_ok = True)
        
        shutil.copy('/mnt/bulk-ganymede/narmin/narmin/AML_Project/AML_Project/evaluate_AML_All_In.py', os.path.join(OUT_DIR, 'code_copy.txt'))
        
        DIAGNOSE_DIR = OUT_DIR + '/diagnose'
        os.makedirs(DIAGNOSE_DIR, exist_ok = True)
        
        patients = get_patient_names(clini_table = TABLE_DIR + '/random_50_patients.csv')

        with open('/mnt/bulk-ganymede/narmin/narmin/MSI_LLM/key.json', 'r') as f:
            config = json.load(f)
        
        client = OpenAI(api_key=config['Pluto'], base_url= 'http://192.168.33.27/v1')
        
        for patient in tqdm(patients):
            
            images = load_images(image_dir = IMG_DIR, patient = patient)
            
            
            encoded_images = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}}
                for path in images
            ]
            
            diagnose = generate_diagnose(encoded_images = encoded_images, client = client, model_name = model_name, max_tokens=2000, temperature=0.2)
            
            diagnose_file = os.path.join(DIAGNOSE_DIR,  patient + '.json')  
            
            entry = {"Patient": patient,"Ergebnis": diagnose}         
                 
            with open(diagnose_file, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
                
        write_results_to_csv(input_dir = DIAGNOSE_DIR, output_file = os.path.join(OUT_DIR, 'results.xlsx'))
    else:
        OUT_DIR = '/mnt/bulk-ganymede/narmin/narmin/AML_Project/output/run_2025-07-18_12-55-00_medgemma-27b-it-q6'
        DIAGNOSE_DIR = OUT_DIR + '/diagnose'
        
        write_results_to_csv(input_dir = DIAGNOSE_DIR,
                             output_file = os.path.join(OUT_DIR, 'results.xlsx'))
            
