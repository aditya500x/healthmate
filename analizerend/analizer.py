import easyocr
import os
import re

# Initialize EasyOCR Reader globally on import.
# This prevents slow re-initialization on every API call.
# Setting gpu=True for acceleration on the user's RTX GPU (requires CUDA setup).
try:
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

def analyze_prescription_image(file_path: str) -> list[str]:
    """
    Runs OCR on the image and uses simple heuristics (rule-based extraction)
    to find medication names from the raw text.
    """
    if READER is None:
        return ["Error: OCR engine failed to load."]

    if not os.path.exists(file_path):
        return ["Error: Input file not found on server."]

    try:
        # 1. OCR Step: Extract text from the image
        results = READER.readtext(file_path)
        
        # Concatenate all recognized text into one string
        raw_text = " ".join([res[1] for res in results])
        print(f"ANALYZEREND: Raw OCR Text: {raw_text}")
        
        # 2. Heuristics Step: Simple rule-based extraction
        # Pattern: Find words starting with a capital letter, followed by 3 or more letters/numbers
        words = re.findall(r'\b[A-Z][a-zA-Z0-9]{3,}\w*\b', raw_text)
        
        # Filter for common non-drug words
        medication_candidates = set()
        # Expanded list of common non-drug words to exclude
        non_drugs = {'Dr', 'Date', 'Name', 'Phone', 'Address', 'Refill', 'Patient', 'Signature', 'Take', 'Dispense', 'Prescription', 'Hospital', 'Clinic', 'Pharmacy', 'Directions', 'Refills', 'Tablets'}
        
        for word in words:
            # Ensure the word isn't entirely capitalized (like ACRONYMS, which might be dosage instructions or acronyms we don't want)
            if word not in non_drugs and not word.isupper():
                 medication_candidates.add(word)

        if not medication_candidates:
            return [f"Could not reliably extract medicine names. Raw Text Snippet: {raw_text[:50]}..."]

        return sorted(list(medication_candidates))

    except Exception as e:
        print(f"ANALYZEREND: Error during analysis: {e}")
        return [f"Critical Analysis Error: {e}"]
