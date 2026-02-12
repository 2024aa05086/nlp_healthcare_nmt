"""
Neural Machine Translation Model for Healthcare Domain
"""

import torch
from transformers import (
    MarianMTModel, MarianTokenizer,
    M2M100ForConditionalGeneration, M2M100Tokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer
)
from typing import List, Dict, Optional, Tuple
import json
import os


class HealthcareNMT:
    """Neural Machine Translation System specialized for Healthcare Domain"""

    def __init__(self, model_name: str = None, config: Dict = None):
        """
        Initialize NMT model

        Args:
            model_name: Name of pretrained model
            config: Configuration dictionary
        """
        if config is None:
            from .utils import load_config
            config = load_config()

        self.config = config
        self.model_name = model_name or config['model']['default_model']
        self.device = self._get_device()

        # Load healthcare terminology
        self.healthcare_terms = self._load_healthcare_terms()

        # Initialize model and tokenizer
        self.model, self.tokenizer = self._load_model()

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded: {self.model_name}")
        print(f"✅ Device: {self.device}")
        if torch.cuda.is_available():
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def _get_device(self) -> torch.device:
        """Get appropriate device (GPU if available)"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_healthcare_terms(self) -> Dict:
        """Load healthcare terminology dictionary"""
        terms_file = self.config['healthcare']['terminology_file']
        if os.path.exists(terms_file):
            from .utils import load_json_file
            return load_json_file(terms_file)
        else:
            # Default healthcare terms
            return {
                "en": {
                    "hypertension": {"fr": "hypertension", "es": "hipertensión", "de": "Hypertonie"},
                    "diabetes": {"fr": "diabète", "es": "diabetes", "de": "Diabetes"},
                    "antibiotic": {"fr": "antibiotique", "es": "antibiótico", "de": "Antibiotikum"},
                    "symptom": {"fr": "symptôme", "es": "síntoma", "de": "Symptom"},
                    "diagnosis": {"fr": "diagnostic", "es": "diagnóstico", "de": "Diagnose"},
                    "treatment": {"fr": "traitement", "es": "tratamiento", "de": "Behandlung"},
                    "prescription": {"fr": "ordonnance", "es": "receta", "de": "Rezept"},
                    "dosage": {"fr": "posologie", "es": "dosificación", "de": "Dosierung"}
                }
            }

    def _load_model(self) -> Tuple:
        """Load appropriate model based on model name"""
        if "opus-mt" in self.model_name.lower():
            tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            model = MarianMTModel.from_pretrained(self.model_name)
        elif "m2m100" in self.model_name.lower():
            tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
        else:
            # Try Auto classes as fallback
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        return model, tokenizer

    def preprocess_medical_text(self, text: str, target_lang: str = None) -> str:
        """
        Preprocess medical text for better translation

        Args:
            text: Input medical text
            target_lang: Target language code (fr, es, de, etc.)

        Returns:
            Preprocessed text
        """
        from .utils import clean_text
        text = clean_text(text)

        # Add explanations for complex medical terms
        if target_lang and target_lang in ['fr', 'es', 'de']:
            for term, translations in self.healthcare_terms.get("en", {}).items():
                if term.lower() in text.lower():
                    if target_lang in translations:
                        # Replace with known translation
                        text = text.replace(term, translations[target_lang])

        return text

    def translate(self,
                  source_text: str,
                  target_lang: str = None,
                  **generation_kwargs) -> str:
        """
        Translate text using the NMT model

        Args:
            source_text: Text to translate
            target_lang: Target language code
            **generation_kwargs: Additional generation parameters

        Returns:
            Translated text
        """
        if not source_text.strip():
            return ""

        # Preprocess medical text
        processed_text = self.preprocess_medical_text(source_text, target_lang)

        # Prepare generation parameters
        gen_config = self.config['model']['generation'].copy()
        gen_config.update(generation_kwargs)

        # For M2M100 models, set target language
        if "m2m100" in self.model_name.lower() and target_lang:
            self.tokenizer.src_lang = "en"
            self.tokenizer.tgt_lang = target_lang

        # Tokenize input
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=gen_config['max_length'],
            padding=True
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=gen_config['max_length'],
                num_beams=gen_config['num_beams'],
                temperature=gen_config['temperature'],
                do_sample=gen_config['do_sample'],
                length_penalty=gen_config['length_penalty'],
                repetition_penalty=gen_config['repetition_penalty'],
                early_stopping=True
            )

        # Decode output
        translation = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return translation

    def batch_translate(self,
                        texts: List[str],
                        target_lang: str = None,
                        batch_size: int = 8,
                        **kwargs) -> List[str]:
        """
        Translate multiple texts in batches

        Args:
            texts: List of texts to translate
            target_lang: Target language code
            batch_size: Batch size for translation
            **kwargs: Additional translation parameters

        Returns:
            List of translations
        """
        translations = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_translations = []

            for text in batch:
                translation = self.translate(text, target_lang, **kwargs)
                batch_translations.append(translation)

            translations.extend(batch_translations)

            # Print progress
            if i + batch_size < len(texts):
                print(f"Translated {i + batch_size}/{len(texts)} texts...")

        return translations

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "model_class": self.model.__class__.__name__,
            "tokenizer_class": self.tokenizer.__class__.__name__
        }