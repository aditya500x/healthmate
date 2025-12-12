import os
import re
import json
import numpy as np
import cv2
# --- CUSTOM OCR IMPORTS ---
from paddleocr import PaddleOCR
from fuzzywuzzy import fuzz, process
# ----------------------------

# --- GLOBAL VARIABLES & CACHE ---
# FIX: Set environment variable to bypass PaddleOCR connectivity check
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
CUSTOM_OCR_READER = None
READER_INITIALIZED = False
# --------------------------------

# --- Initialization Function (Moved outside global scope) ---
def initialize_ocr_reader():
    """
    Initializes and caches the PaddleOCR reader. 
    FIX: Removes the 'use_gpu' argument to prevent 'Unknown argument' error.
    """
    global CUSTOM_OCR_READER, READER_INITIALIZED
    if READER_INITIALIZED:
        return CUSTOM_OCR_READER
        
    try:
        # PaddleOCR will now perform automatic detection (defaults to CPU if GPU fails or is not configured)
        CUSTOM_OCR_READER = PaddleOCR(lang='en') 
        print("ANALYZEREND: Custom PaddleOCR initialized (Auto GPU/CPU detection - Arguments fixed).")
        
    except Exception as e:
        print(f"ANALYZEREND: CRITICAL ERROR: PaddleOCR initialization failed: {e}. Check dependencies (PaddleOCR, cv2).")
        CUSTOM_OCR_READER = None
            
    READER_INITIALIZED = True
    return CUSTOM_OCR_READER


# --- CUSTOM MEDICAL DICTIONARY (Copied from enhanced_ocr.py) ---
MEDICATION_DICT = {
    # Common medications - Updated for Indian market
    "amoxicillin": ["amox", "amoxil", "amoxicil", "amoxicilin", "mox", "novamox", "almoxi", "wymox"],
    "paracetamol": ["paracet", "parcetamol", "acetaminophen", "tylenol", "crocin", "panadol", "dolo", "metacin", "calpol", "sumo", "febrex", "acepar", "pacimol"],
    "ibuprofen": ["ibuprofin", "ibu", "ibuprofen", "advil", "motrin", "nurofen", "brufen", "ibugesic", "combiflam"],
    "aspirin": ["asa", "acetylsalicylic", "aspr", "disprin", "ecotrin", "bayer", "loprin", "delisprin", "colsprin"],
    "lisinopril": ["lisin", "prinivil", "zestril", "qbrelis", "listril", "hipril", "zestopril"],
    "metformin": ["metform", "glucophage", "fortamet", "glumetza", "riomet", "glycomet", "obimet", "gluformin", "glyciphage"],
    "atorvastatin": ["lipitor", "atorva", "atorvastat", "lipibec", "atorlip", "atocor", "storvas"],
    "omeprazole": ["prilosec", "omepraz", "losec", "zegerid", "priosec", "omez", "ocid", "prazole"],
    "amlodipine": ["norvasc", "amlo", "amlod", "katerzia", "norvasc", "amlopress", "amlopres", "amlokind"],
    "levothyroxine": ["synthroid", "levothy", "levothyrox", "levoxyl", "tirosint", "euthyrox", "thyronorm", "eltroxin"],
    "metoprolol": ["lopressor", "toprol", "metopro", "toprol-xl", "betaloc", "metolar", "starpress"],
    "sertraline": ["zoloft", "sert", "sertra", "lustral", "serta", "daxid", "serlin"],
    "gabapentin": ["neurontin", "gaba", "gabap", "gralise", "horizant", "gabapin", "gaban", "progaba"],
    "hydrochlorothiazide": ["hctz", "hydrochlor", "microzide", "hydrodiuril", "hydrazide", "aquazide"],
    "simvastatin": ["zocor", "simvast", "simlup", "simcard", "simvotin", "zosta", "simgal"],
    "losartan": ["cozaar", "losart", "lavestra", "repace", "losar", "zaart", "covance"],
    "albuterol": ["proventil", "ventolin", "proair", "salbutamol", "asthalin", "ventofort", "aeromist"],
    "fluoxetine": ["prozac", "sarafem", "rapiflux", "prodep", "fludac", "flunil", "flunat"],
    "citalopram": ["celexa", "cipramil", "citalo", "celepram", "citalex", "citopam"],
    "pantoprazole": ["protonix", "pantoloc", "pantocid", "pantodac", "zipant", "pan"],
    "furosemide": ["lasix", "furos", "frusemide", "frusol", "lasix", "frusenex", "diucontin"],
    "rosuvastatin": ["crestor", "rosuvast", "rosuvas", "rovista", "rostor", "colver"],
    "escitalopram": ["lexapro", "cipralex", "nexito", "feliz", "stalopam", "nexito-forte"],
    "montelukast": ["singulair", "montek", "montair", "monticope", "romilast", "monty-lc"],
    "prednisone": ["deltasone", "predni", "orasone", "predcip", "omnacortil", "wysolone"],
    "warfarin": ["coumadin", "jantoven", "warf", "warfex", "warfrant", "uniwarfin"],
    "tramadol": ["ultram", "tram", "tramahexal", "tramazac", "domadol", "tramacip"],
    "azithromycin": ["zithromax", "azithro", "z-pak", "azith", "azee", "aziwok", "azimax", "zithrocin"],
    "ciprofloxacin": ["cipro", "ciloxan", "ciproxin", "ciplox", "ciprobid", "cifran", "ciprinol"],
    "lamotrigine": ["lamictal", "lamot", "lamotrigin", "lamogard", "lamitor", "lametec"],
    "venlafaxine": ["effexor", "venlaf", "venlor", "veniz", "ventab", "venlift"],
    "insulin": ["lantus", "humulin", "novolin", "humalog", "novolog", "tresiba", "insugen", "wosulin", "basalog", "insuman", "apidra"],
    "metronidazole": ["flagyl", "metro", "metrogel", "metrogyl", "metrozole", "aristogyl"],
    "naproxen": ["aleve", "naprosyn", "anaprox", "xenar", "napra", "napxen"],
    "doxycycline": ["vibramycin", "oracea", "doxy", "doxin", "biodoxi", "doxt"],
    "cetirizine": ["zyrtec", "cetryn", "cetriz", "alerid", "cetcip", "zirtin", "cetzine"],
    "diazepam": ["valium", "valpam", "dizac", "calmpose", "zepose", "sedopam"],
    "alprazolam": ["xanax", "alprax", "tafil", "alp", "alzolam", "zolax", "restyl", "trika"],
    "clonazepam": ["klonopin", "rivotril", "clon", "petril", "clonopam", "lonazep"],
    "carvedilol": ["coreg", "carvedil", "cardivas", "carca", "carvil", "carloc"],
    "fexofenadine": ["allegra", "telfast", "fexofine", "fexova", "agimfast", "allerfast"],
    "ranitidine": ["zantac", "ranit", "rantec", "aciloc", "zinetac", "histac"],
    "diclofenac": ["voltaren", "diclof", "diclomax", "voveran", "diclonac", "reactin"],
    "ceftriaxone": ["rocephin", "ceftri", "cefaxone", "inocef", "trixone", "monotax"],
    "cefixime": ["suprax", "cefi", "taxim", "unice", "cefispan", "omnicef"],
    "esomeprazole": ["nexium", "esotrex", "esopral", "nexpro", "raciper", "sompraz"],
    "clopidogrel": ["plavix", "clopid", "plagerine", "clopilet", "deplatt", "noklot"],
    "levocetirizine": ["xyzal", "levocet", "teczine", "levazeo", "xyzra", "uvnil"],
    "febuxostat": ["uloric", "febugat", "febuget", "zylobact", "febustat"],
    "telmisartan": ["micardis", "telma", "telsar", "sartel", "telvas", "cresar"],
    "folic acid": ["folate", "folvite", "folet", "folacin", "folitab", "obifolic"],
    "olmesartan": ["olmat", "olmy", "benitec", "olmezest", "olsar"],
    "vildagliptin": ["galvus", "zomelis", "jalra", "vysov", "vildalip", "viladay"],
    "sitagliptin": ["januvia", "sitagen", "istamet", "janumet", "sitaglip", "trevia"],
    "glimepiride": ["amaryl", "glimpid", "glymex", "zoryl", "glimer", "diaglip"],
    "gliclazide": ["diamicron", "lycazid", "glizid", "reclide", "odinase", "glynase"],
    "ramipril": ["altace", "cardiopril", "cardace", "ramiril", "celapres", "ramace"],
    "nebivolol": ["bystolic", "nebistar", "nebicard", "nebilong", "nebilet", "nubeta"],
    "cilnidipine": ["cilacar", "cinod", "ciladay", "neudipine", "cilaheart", "cidip"],
    "rabeprazole": ["aciphex", "rablet", "rabicip", "razo", "raboz", "pepcia"],
    "dexamethasone": ["decadron", "dexona", "dexamycin", "dexacort", "decilone", "dexasone"],
    "doxofylline": ["doxolin", "synasma", "doxobid", "doxovent", "doxoril", "doxfree"],
    "deflazacort": ["dezacor", "defcort", "defza", "xenocort", "flacort", "defolet"],
    "ondansetron": ["zofran", "ondem", "emeset", "vomitrol", "zondem", "ondemet"],
    "domperidone": ["motilium", "domstal", "vomistop", "dompan", "dompy", "domcolic"],
    "pantoprazole": ["pantocid", "pantop", "pantodac", "panto", "pan-d", "pantozol"],
    "mefenamic acid": ["ponstan", "meftal", "meflam", "mefkind", "rafen", "mefgesic"],
    "aceclofenac": ["aceclo", "hifenac", "zerodol", "movon", "aceclo-plus", "acebid"],
    "nimesulide": ["nise", "nimulid", "nimek", "nimica", "nimcet", "nimsaid"],
    "hydroxyzine": ["atarax", "anxnil", "hyzox", "hydryllin", "anxipar", "hydrax"],
    "amlodipine + atenolol": ["amlokind-at", "amtas-at", "stamlo-beta", "tenolam", "amlopress-at"],
    "telmisartan + hydrochlorothiazide": ["telma-h", "telsar-h", "telvas-h", "tazloc-h", "telista-h"],
    "sulfamethoxazole + trimethoprim": ["bactrim", "septran", "cotrim", "sepmax", "oriprim"],
    "amoxicillin + clavulanic acid": ["augmentin", "moxclav", "megaclox", "clavam", "hiclav", "clavum"],
    "ofloxacin": ["oflox", "oflin", "tarivid", "zenflox", "oflacin", "exocin"],
    "torsemide": ["demadex", "dytor", "tide", "torlactone", "presage", "tomide"],
    "chlorthalidone": ["thalitone", "clorpres", "cloress", "natrilix", "thaloride"],
    "ivermectin": ["stromectol", "ivermect", "ivecop", "ivepred", "scabo", "ivernex"],
    "rifaximin": ["xifaxan", "rifagut", "rcifax", "rifakem", "rifamide"],
    "nitrofurantoin": ["furadantin", "niftran", "nitrofur", "furadoine", "nidantin"],
    "betahistine": ["serc", "vertin", "betaserc", "vertigo", "beta", "histiwel"],
    "etizolam": ["etilaam", "etizola", "sedekopan", "etizaa", "etzee", "etova"],
    "clotrimazole": ["candid", "clotri", "mycomax", "candiderma", "candifun", "clotop"],
    "ketoconazole": ["nizoral", "sebizole", "ketoz", "fungicide", "ketomac", "ketostar"],
    "fluconazole": ["diflucan", "flucz", "forcan", "syscan", "zocon", "flucos"],
    "pregabalin": ["lyrica", "pregeb", "maxgalin", "nervalin", "pregastar", "pregica"],
    "methylprednisolone": ["medrol", "methylpred", "depo-medrol", "solu-medrol", "depopred", "medrate"],
    "levetiracetam": ["keppra", "levesam", "levroxa", "levipil", "levecetam", "epictal"],
}

# Standardized list of known drug names and aliases for case-insensitive matching
KNOWN_DRUGS = set()
for key, aliases in MEDICATION_DICT.items():
    KNOWN_DRUGS.add(key)
    for alias in aliases:
        KNOWN_DRUGS.add(alias)
# --- End Custom Medical Dictionary ---


# --- MOCK INTERACTION DATABASE (Keys must be lowercase) ---
MOCK_INTERACTIONS = {
    'ibuprofen-lisinopril': 'Major interaction: Ibuprofen can reduce the effectiveness of Lisinopril for blood pressure control.',
    'amoxicillin-aspirin': 'Minor interaction: May increase the risk of stomach irritation.',
    'statin-grapefruit': 'Major interaction: Statins (e.g., Atorvastatin) combined with grapefruit can dangerously increase drug levels.',
    'metformin-alcohol': 'Moderate interaction: Alcohol consumption can increase the risk of lactic acidosis with Metformin.',
}
# --- End Mock Interaction Database ---


# --- HELPER FUNCTIONS (Copied/Adapted from your enhanced_ocr.py) ---

def extract_text_paddle_single_pass(image_path):
    """Extract text using PaddleOCR on a single image pass (preprocessed)."""
    # Initialize reader if not already cached
    reader = initialize_ocr_reader()
    if reader is None:
        return "OCR Engine Failed Initialization.", 0.0
    
    try:
        result = reader.ocr(image_path)
        
        text = ""
        confidence_scores = []
        
        # Handle PaddleOCR result format
        if result and isinstance(result, list) and len(result) > 0 and result[0] is not None:
            for line in result[0]:
                if len(line) >= 2 and isinstance(line[1], list) and len(line[1]) == 2:
                    text += line[1][0] + "\n"
                    confidence_scores.append(float(line[1][1]))
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return text.strip(), avg_confidence * 100 # Return confidence as percentage
        
    except Exception as e:
        print(f"PaddleOCR extraction error: {str(e)}")
        # Return a custom error message for the frontend
        return f"PaddleOCR Error: {str(e)}", 0.0

def apply_medical_dictionary_correction(text):
    """Apply medication name correction using fuzzy matching against KNOWN_DRUGS."""
    if not text:
        return text
    
    words = re.findall(r'\b\w+\b', text.lower())
    corrected_text = text
    
    # Check if fuzzywuzzy is available
    try:
         from fuzzywuzzy import process
    except ImportError:
         # If dependency is missing, skip the correction but don't crash
         return text
    
    for word in set(words):
        if len(word) < 4 or word.isdigit(): continue
            
        match_result = process.extractOne(word, list(KNOWN_DRUGS), scorer=fuzz.ratio)

        if match_result and match_result[1] > 75: # Threshold 75%
            correct_term = match_result[0]
            
            # Replace the OCR output word (case-insensitively) with the correct term
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            # Use 'correct_term' in the final output string
            corrected_text = pattern.sub(correct_term, corrected_text, 1)
            
    return corrected_text

def extract_medications_from_text(text):
    """Simple dictionary lookup and extraction of medications from corrected text."""
    medications = set()
    text_lower = text.lower()

    # Find medications using dictionary keys and aliases
    for key, aliases in MEDICATION_DICT.items():
        search_terms = [key] + aliases
        for term in search_terms:
            # Look for word boundary match
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                medications.add(key.capitalize()) # Add standardized, capitalized name
                break
                
    return list(medications)

def check_drug_interactions(medications: list[str]) -> list[str]:
    """Checks the list of extracted medications against the mock interaction database."""
    warnings = []
    meds_lower = [m.lower() for m in medications]
    
    # 1. Check known two-drug interactions (by forming a standardized key)
    for i in range(len(meds_lower)):
        for j in range(i + 1, len(meds_lower)):
            med1 = meds_lower[i]
            med2 = meds_lower[j]
            
            # Standardize key (alphabetical order)
            interaction_key = '-'.join(sorted([med1, med2]))
            
            if interaction_key in MOCK_INTERACTIONS:
                warnings.append(f"Interaction ({med1.capitalize()} + {med2.capitalize()}): {MOCK_INTERACTIONS[interaction_key]}")

    # 2. Check general warnings (e.g., drug-food interaction)
    for med in meds_lower:
        if 'statin' in med and any(keyword in meds_lower for keyword in ['grapefruit', 'juice']):
            warnings.append(f"Major Alert: {MOCK_INTERACTIONS['statin-grapefruit']}")
        if 'metformin' in med:
             warnings.append(f"General Warning: {MOCK_INTERACTIONS['metformin-alcohol']}")

    return warnings


# --- MAIN ANALYZER FUNCTION ---

def analyze_prescription_image(file_path: str) -> dict:
    """
    Runs Custom OCR, applies dictionary correction, and extracts medications.
    """
    results_dict = {
        "medications": [],
        "interactions": [],
        "raw_text_snippet": "Analysis Failed.",
        "accuracy_score": 0.0
    }

    if not os.path.exists(file_path):
        results_dict["medications"] = ["Error: Input file not found on server."]
        return results_dict

    try:
        # 1. OCR Step: Extract text from the image
        # This will call the robust initialization
        raw_text, confidence = extract_text_paddle_single_pass(file_path)
        
        # Check if initialization/OCR failed
        if raw_text.startswith("OCR Engine Failed") or raw_text.startswith("PaddleOCR Error"):
            results_dict["medications"] = [raw_text]
            results_dict["accuracy_score"] = 0.0
            return results_dict
        
        results_dict["raw_text_snippet"] = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
        results_dict["accuracy_score"] = round(confidence, 1)

        print(f"ANALYZEREND: Raw OCR Text:\n{raw_text}")
        
        # 2. Apply Dictionary Correction (Fuzzy Matching)
        corrected_text = apply_medical_dictionary_correction(raw_text)
        
        # 3. Final Extraction using standardized list lookup
        medications = extract_medications_from_text(corrected_text)
        
        # --- Finalizing Results ---
        if not medications:
            results_dict["medications"] = [f"Could not extract medicine names. Snippet: {results_dict['raw_text_snippet']}"]
            # Default low score for failure
            if confidence < 70:
                results_dict["accuracy_score"] = 35.0
        else:
            results_dict["medications"] = sorted(medications)

        # 4. Check Interactions
        if results_dict["medications"] and not results_dict["medications"][0].startswith("Could not extract"):
            results_dict["interactions"] = check_drug_interactions(results_dict["medications"])
        
        return results_dict

    except Exception as e:
        print(f"ANALYZEREND: Error during analysis: {e}")
        results_dict["medications"] = [f"Critical Analysis Error: {e}"]
        return results_dict
