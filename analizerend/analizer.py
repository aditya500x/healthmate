import easyocr
import os
import re

# Initialize EasyOCR Reader globally on import.
try:
    # Setting gpu=True for acceleration on the user's RTX GPU (requires CUDA setup).
    READER = easyocr.Reader(['en'], gpu=True) 
    print("ANALYZEREND: EasyOCR reader initialized (GPU enabled).")
except Exception as e:
    # Fallback to CPU if CUDA/GPU setup fails
    print(f"ANALYZEREND: WARNING: GPU initialization failed ({e}). Falling back to CPU.")
    try:
        READER = easyocr.Reader(['en'], gpu=False)
        print("ANALYZEREND: EasyOCR reader initialized (CPU fallback).")
    except Exception as fallback_e:
        print(f"ANALYZEREND: CRITICAL ERROR: CPU initialization failed too: {fallback_e}")
        READER = None

# --- Advanced Heuristics Configuration & Mock Database ---
# Common endings for pharmaceutical names and treatments
DRUG_SUFFIXES = [
    'pril', 'sartan', 'olol', 'statin', 'zepam', 'prazole', 'fenac', 'vir',
    'cin', 'cillin', 'mycin', 'tine', 'amine', 'done', 'mol', 'rol',
    'cetam', 'xetine', 'nidazole', 'barbital', 'profen',
    # Added common solution/treatment names and forms (glucose/dextrose)
    'dextrose', 'solution', 'fluid', 'intake', 'ors', 'iv', 'ip',
    # Common medical terms
    'syrup', 'drops', 'ointment', 'cream', 'tablet', 'cap'
]
# Common prescription abbreviations and words to ignore (all lowercase for matching)
NON_DRUG_WORDS = {
    'dr', 'date', 'name', 'phone', 'address', 'refill', 'patient', 'signature', 
    'take', 'dispense', 'prescription', 'hospital', 'clinic', 'pharmacy', 
    'directions', 'refills', 'tablets', 'capsules', 'lot', 'qty', 'ref',
    'institute', 'sciences', 'research', 'centre', 'uhid', 'no', 'pr', 'bp', 'bs', 
    'clo', 'giddiness', 'restlessness', 'adv', 'adequate', 'bpm', 'trtp'
}

# MOCK INTERACTION DATABASE (Keys must be lowercase)
MOCK_INTERACTIONS = {
    'ibuprofen-lisinopril': 'Major interaction: Ibuprofen can reduce the effectiveness of Lisinopril for blood pressure control.',
    'amoxicillin-aspirin': 'Minor interaction: May increase the risk of stomach irritation.',
    'statin-grapefruit': 'Major interaction: Statins (e.g., Atorvastatin) combined with grapefruit can dangerously increase drug levels.',
    'metformin-alcohol': 'Moderate interaction: Alcohol consumption can increase the risk of lactic acidosis with Metformin.',
}
# --- End Heuristics Configuration & Mock Database ---

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


def analyze_prescription_image(file_path: str) -> dict:
    """
    Runs OCR and heuristics, filtering text based on drug suffixes, ignoring case.
    """
    results_dict = {
        "medications": [],
        "interactions": [],
        "raw_text_snippet": "Analysis Failed.",
        "accuracy_score": 0.0
    }

    if READER is None or not os.path.exists(file_path):
        results_dict["medications"] = ["Error: OCR engine failed to load or file not found."]
        return results_dict

    try:
        # 1. OCR Step: Extract text from the image
        results = READER.readtext(file_path)
        raw_text = "\n".join([res[1] for res in results])
        raw_text_lower = raw_text.lower() # Convert entire text to lowercase
        
        results_dict["raw_text_snippet"] = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
        print(f"ANALYZEREND: Raw OCR Text (Lower):\n{raw_text_lower}")
        
        medication_candidates = set()
        
        # 2. Heuristics Extraction: Case-insensitive word search
        # Find all words (min 2 chars) in the text
        all_words = set(re.findall(r'\b[a-zA-Z0-9]{2,}\w*\b', raw_text_lower)) 
            
        total_words = 0
        matched_drug_keywords = 0
            
        for word_lower in all_words:
            cleaned_word = re.sub(r'[^a-z0-9]+$', '', word_lower)
            
            # Skip single characters and noise
            if len(cleaned_word) <= 1:
                continue

            total_words += 1

            # 2a. Check against known non-drug words
            if cleaned_word in NON_DRUG_WORDS:
                continue
            
            # 2b. Check if the word ends with a common drug suffix
            is_drug_candidate = False
            for suffix in DRUG_SUFFIXES:
                # Match full word (for acronyms like ORS) or word ending
                if cleaned_word.endswith(suffix) or cleaned_word == suffix:
                    is_drug_candidate = True
                    break
            
            if is_drug_candidate:
                matched_drug_keywords += 1
                # Add the word to candidates, capitalized for display
                medication_candidates.add(cleaned_word.capitalize())

        # 3. Calculate Accuracy Score (Pseudo-Accuracy)
        if total_words > 0:
            # Score based on how dense the matching keywords are compared to all words
            score = (matched_drug_keywords / total_words) * 100 * 2 # Adjusted ratio, scaled for more realistic score
            results_dict["accuracy_score"] = min(99.9, round(score, 1))
        
        # --- Finalizing Results ---
        if not medication_candidates:
            results_dict["medications"] = [f"Could not extract medicine names. Snippet: {results_dict['raw_text_snippet']}"]
            # Default low score for failure
            results_dict["accuracy_score"] = 35.0 
        else:
            results_dict["medications"] = sorted(list(medication_candidates))

        # 4. Check Interactions
        if results_dict["medications"] and not results_dict["medications"][0].startswith("Could not extract"):
            results_dict["interactions"] = check_drug_interactions(results_dict["medications"])
        
        return results_dict

    except Exception as e:
        print(f"ANALYZEREND: Error during analysis: {e}")
        results_dict["medications"] = [f"Critical Analysis Error: {e}"]
        return results_dict
