"""
Gradio User Interface for Healthcare NMT System
"""

import gradio as gr
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
import os

from .nmt_model import HealthcareNMT
from .bleu_evaluator import BLEUEvaluator
from .utils import load_config, load_json_file


class NMTApp:
    """Gradio Interface for Healthcare NMT System"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize NMT Application

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.nmt = None
        self.evaluator = BLEUEvaluator(self.config)
        self.translation_history = []

        # Load sample texts
        self.sample_texts = self._load_sample_texts()

        # Initialize model with default
        self.current_model = self.config['model']['default_model']
        self.nmt = HealthcareNMT(self.current_model, self.config)

        print("✅ Healthcare NMT Application initialized")

    def _load_sample_texts(self) -> Dict:
        """Load sample medical texts"""
        sample_file = self.config['healthcare']['sample_texts_file']
        if os.path.exists(sample_file):
            try:
                return load_json_file(sample_file)
            except:
                pass

        # Default sample texts
        return {
            "Medical Instructions": {
                "text": "Take one tablet twice daily with food. Do not exceed the recommended dosage.",
                "references": {
                    "fr": [
                        "Prenez un comprimé deux fois par jour avec de la nourriture. Ne dépassez pas la dose recommandée."],
                    "es": ["Tome una tableta dos veces al día con alimentos. No exceda la dosis recomendada."],
                    "de": [
                        "Nehmen Sie eine Tablette zweimal täglich mit Nahrung ein. Überschreiten Sie nicht die empfohlene Dosierung."]
                }
            },
            "Diagnosis Report": {
                "text": "Patient presents with symptoms of acute bronchitis, including persistent cough and wheezing.",
                "references": {
                    "fr": [
                        "Le patient présente des symptômes de bronchite aiguë, notamment une toux persistante et une respiration sifflante."],
                    "es": [
                        "El paciente presenta síntomas de bronquitis aguda, incluyendo tos persistente y sibilancias."],
                    "de": [
                        "Der Patient zeigt Symptome einer akuten Bronchitis, einschließlich anhaltendem Husten und Keuchen."]
                }
            },
            "Prescription": {
                "text": "Prescribe amoxicillin 500mg three times daily for seven days to treat bacterial infection.",
                "references": {
                    "fr": [
                        "Prescrire de l'amoxicilline 500 mg trois fois par jour pendant sept jours pour traiter l'infection bactérienne."],
                    "es": [
                        "Recetar amoxicilina 500 mg tres veces al día durante siete días para tratar la infección bacteriana."],
                    "de": [
                        "Verschreiben Sie Amoxicillin 500 mg dreimal täglich für sieben Tage zur Behandlung der bakteriellen Infektion."]
                }
            }
        }

    def get_language_from_model(self, model_name: str) -> str:
        """Extract target language from model name"""
        if "en-fr" in model_name:
            return "fr"
        elif "en-es" in model_name:
            return "es"
        elif "en-de" in model_name:
            return "de"
        else:
            return "en"  # Default

    def translate_and_evaluate(self,
                               source_text: str,
                               model_name: str,
                               reference_text: str,
                               generation_params: Dict = None) -> Tuple:
        """
        Translate text and evaluate with BLEU

        Args:
            source_text: Source text to translate
            model_name: Model to use
            reference_text: Reference translations (one per line)
            generation_params: Generation parameters

        Returns:
            Tuple of results
        """
        if not source_text.strip():
            return self._get_empty_results()

        # Update model if changed
        if model_name != self.current_model:
            self.nmt = HealthcareNMT(model_name, self.config)
            self.current_model = model_name

        # Get target language
        target_lang = self.get_language_from_model(model_name)

        # Prepare generation parameters
        if generation_params is None:
            generation_params = {}

        # Translate
        try:
            translation = self.nmt.translate(
                source_text,
                target_lang=target_lang,
                **generation_params
            )
        except Exception as e:
            translation = f"Translation error: {str(e)}"
            return self._get_error_results(translation)

        # Prepare references
        references = [ref.strip() for ref in reference_text.split('\n') if ref.strip()]

        # Calculate BLEU if references provided
        bleu_result = {}
        analysis_data = []
        history_df = pd.DataFrame()

        if references:
            bleu_result = self.evaluator.calculate_bleu(translation, references, detailed=True)

            # Create analysis table
            for n in range(1, 5):
                precision_info = bleu_result.get("ngram_precisions", {}).get(n, {})
                precision = precision_info.get("precision", 0)
                interpretation = self._interpret_precision(precision)
                analysis_data.append([
                    f"{n}-gram",
                    f"{precision:.2f}%",
                    interpretation,
                    precision_info.get("total_ngrams", 0),
                    precision_info.get("clipped_matches", 0)
                ])

        # Add to history
        self.translation_history.append({
            "Source": source_text[:50] + "..." if len(source_text) > 50 else source_text,
            "Translation": translation[:50] + "..." if len(translation) > 50 else translation,
            "BLEU Score": bleu_result.get("bleu_score", 0),
            "Model": model_name.split("/")[-1]
        })

        # Keep only last 20 entries
        if len(self.translation_history) > 20:
            self.translation_history = self.translation_history[-20:]

        # Prepare history dataframe
        if self.translation_history:
            history_df = pd.DataFrame(self.translation_history)

        # Prepare quality analysis
        quality_info = bleu_result.get("interpretation", {})

        return (
            translation,
            bleu_result.get("bleu_score", 0),
            bleu_result.get("brevity_penalty", 1.0),
            bleu_result.get("ngram_precisions", {}).get(1, {}).get("precision", 0),
            bleu_result.get("ngram_precisions", {}).get(2, {}).get("precision", 0),
            bleu_result.get("ngram_precisions", {}).get(3, {}).get("precision", 0),
            bleu_result.get("ngram_precisions", {}).get(4, {}).get("precision", 0),
            analysis_data,
            history_df,
            quality_info.get("quality", "N/A"),
            quality_info.get("description", "N/A")
        )

    def _interpret_precision(self, precision: float) -> str:
        """Interpret n-gram precision score"""
        if precision >= 90:
            return "Excellent"
        elif precision >= 75:
            return "Very Good"
        elif precision >= 60:
            return "Good"
        elif precision >= 45:
            return "Fair"
        elif precision >= 30:
            return "Poor"
        else:
            return "Very Poor"

    def _get_empty_results(self) -> Tuple:
        """Return empty results"""
        return ("", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, [], pd.DataFrame(), "N/A", "N/A")

    def _get_error_results(self, error_msg: str) -> Tuple:
        """Return error results"""
        return (error_msg, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, [], pd.DataFrame(), "Error", error_msg)

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(
                title="Healthcare NMT with BLEU Evaluation",
                theme=gr.themes.Soft(),
                css=".gradio-container {max-width: 1200px !important;}"
        ) as app:
            # Header
            gr.Markdown("""
            # 🏥 Healthcare Neural Machine Translation System
            ### Translate medical texts with automatic BLEU score evaluation
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    with gr.Group():
                        gr.Markdown("### 📝 Input Medical Text")
                        source_text = gr.Textbox(
                            label="Enter medical text to translate",
                            placeholder="Example: The patient requires immediate attention for hypertension...",
                            lines=5,
                            elem_id="source_text"
                        )

                    # Model selection
                    with gr.Group():
                        gr.Markdown("### 🤖 Translation Model")
                        model_choice = gr.Dropdown(
                            choices=self.config['model']['available_models'],
                            value=self.config['model']['default_model'],
                            label="Select translation model",
                            elem_id="model_choice"
                        )

                    # Reference translation
                    with gr.Group():
                        gr.Markdown("### 📋 Reference Translation")
                        gr.Markdown("Enter reference translations for BLEU evaluation (one per line)")
                        reference_text = gr.Textbox(
                            label="Reference translations",
                            placeholder="Paste reference translations here...",
                            lines=3,
                            elem_id="reference_text"
                        )

                    # Sample texts
                    with gr.Group():
                        gr.Markdown("### 🧪 Sample Medical Texts")
                        sample_names = list(self.sample_texts.keys())
                        sample_dropdown = gr.Dropdown(
                            choices=sample_names,
                            label="Load sample text",
                            value=sample_names[0] if sample_names else None,
                            elem_id="sample_dropdown"
                        )

                        def load_sample(sample_name):
                            if sample_name in self.sample_texts:
                                text = self.sample_texts[sample_name]["text"]
                                model = self.config['model']['default_model']
                                target_lang = self.get_language_from_model(model)

                                # Get reference for current model language
                                refs = self.sample_texts[sample_name]["references"].get(
                                    target_lang, [""]
                                )
                                reference = "\n".join(refs)

                                return text, reference
                            return "", ""

                        sample_dropdown.change(
                            load_sample,
                            inputs=[sample_dropdown],
                            outputs=[source_text, reference_text]
                        )

                    # Action buttons
                    with gr.Row():
                        translate_btn = gr.Button(
                            "🚀 Translate & Evaluate",
                            variant="primary",
                            size="lg"
                        )
                        clear_btn = gr.Button(
                            "🔄 Clear All",
                            variant="secondary"
                        )

                with gr.Column(scale=3):
                    # Results section
                    with gr.Tabs():
                        with gr.TabItem("📄 Translation Output"):
                            gr.Markdown("### NMT Translation Result")
                            translation_output = gr.Textbox(
                                label="Translated Text",
                                interactive=False,
                                lines=4,
                                elem_id="translation_output"
                            )

                        with gr.TabItem("📊 BLEU Evaluation"):
                            with gr.Group():
                                gr.Markdown("### BLEU Score Analysis")

                                with gr.Row():
                                    bleu_score = gr.Number(
                                        label="BLEU Score",
                                        precision=2,
                                        elem_id="bleu_score"
                                    )
                                    brevity_penalty = gr.Number(
                                        label="Brevity Penalty",
                                        precision=3,
                                        elem_id="brevity_penalty"
                                    )

                                gr.Markdown("#### N-gram Precisions")
                                with gr.Row():
                                    unigram_precision = gr.Number(
                                        label="1-gram Precision",
                                        precision=2,
                                        elem_id="unigram_precision"
                                    )
                                    bigram_precision = gr.Number(
                                        label="2-gram Precision",
                                        precision=2,
                                        elem_id="bigram_precision"
                                    )
                                    trigram_precision = gr.Number(
                                        label="3-gram Precision",
                                        precision=2,
                                        elem_id="trigram_precision"
                                    )
                                    fourgram_precision = gr.Number(
                                        label="4-gram Precision",
                                        precision=2,
                                        elem_id="fourgram_precision"
                                    )

                        with gr.TabItem("🔍 Detailed Analysis"):
                            gr.Markdown("### Detailed N-gram Analysis")
                            analysis_output = gr.DataFrame(
                                label="N-gram Precision Breakdown",
                                headers=["N-gram", "Precision", "Quality", "Total N-grams", "Matches"],
                                interactive=False,
                                elem_id="analysis_output"
                            )

                            gr.Markdown("### Quality Assessment")
                            with gr.Row():
                                quality_label = gr.Textbox(
                                    label="Quality Level",
                                    interactive=False
                                )
                                quality_description = gr.Textbox(
                                    label="Description",
                                    interactive=False
                                )

                        with gr.TabItem("📜 Translation History"):
                            gr.Markdown("### Recent Translations")
                            history_df = gr.DataFrame(
                                label="Translation History",
                                interactive=False,
                                elem_id="history_df"
                            )

            # Multiple evaluation section
            gr.Markdown("## 📈 Multiple Translation Evaluation")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Compare multiple candidate translations")
                    multi_candidates = gr.Textbox(
                        label="Candidate Translations (one per line)",
                        placeholder="Enter candidate translations to compare...",
                        lines=4,
                        elem_id="multi_candidates"
                    )

                    multi_references = gr.Textbox(
                        label="Reference Translations (one per line per candidate)",
                        placeholder="Enter corresponding references...",
                        lines=4,
                        elem_id="multi_references"
                    )

                    multi_eval_btn = gr.Button(
                        "📊 Evaluate Multiple",
                        variant="primary"
                    )

                with gr.Column():
                    multi_results = gr.DataFrame(
                        label="Multiple Evaluation Results",
                        interactive=False,
                        elem_id="multi_results"
                    )

            # Generation parameters (advanced)
            with gr.Accordion("⚙️ Advanced Generation Parameters", open=False):
                with gr.Row():
                    num_beams = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=self.config['model']['generation']['num_beams'],
                        step=1,
                        label="Number of beams"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=self.config['model']['generation']['temperature'],
                        step=0.1,
                        label="Temperature"
                    )
                    length_penalty = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=self.config['model']['generation']['length_penalty'],
                        step=0.1,
                        label="Length penalty"
                    )

            # Callback functions
            def translate_callback(text, model, reference, beams, temp, length_pen):
                generation_params = {
                    "num_beams": int(beams),
                    "temperature": temp,
                    "length_penalty": length_pen
                }
                return self.translate_and_evaluate(text, model, reference, generation_params)

            def evaluate_multiple_callback(candidates_text, references_text):
                if not candidates_text.strip() or not references_text.strip():
                    return pd.DataFrame()

                candidates = [c.strip() for c in candidates_text.split('\n') if c.strip()]
                references = [r.strip() for r in references_text.split('\n') if r.strip()]

                # Group references (one per candidate for simplicity)
                references_list = [[ref] for ref in references[:len(candidates)]]

                # If fewer references than candidates, repeat last reference
                if len(references_list) < len(candidates):
                    last_ref = references_list[-1] if references_list else [""]
                    while len(references_list) < len(candidates):
                        references_list.append(last_ref)

                results = self.evaluator.evaluate_multiple(candidates, references_list)
                return results

            def clear_callback():
                return ("", "", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, [], pd.DataFrame(), "N/A", "N/A",
                        "", "", pd.DataFrame())

            # Connect callbacks
            translate_btn.click(
                translate_callback,
                inputs=[source_text, model_choice, reference_text, num_beams, temperature, length_penalty],
                outputs=[
                    translation_output,
                    bleu_score,
                    brevity_penalty,
                    unigram_precision,
                    bigram_precision,
                    trigram_precision,
                    fourgram_precision,
                    analysis_output,
                    history_df,
                    quality_label,
                    quality_description
                ]
            )

            multi_eval_btn.click(
                evaluate_multiple_callback,
                inputs=[multi_candidates, multi_references],
                outputs=[multi_results]
            )

            clear_btn.click(
                clear_callback,
                outputs=[
                    source_text, reference_text, bleu_score, brevity_penalty,
                    unigram_precision, bigram_precision, trigram_precision,
                    fourgram_precision, analysis_output, history_df,
                    quality_label, quality_description,
                    multi_candidates, multi_references, multi_results
                ]
            )

            # Footer with instructions
            gr.Markdown("""
            ---
            ## 📖 Instructions

            1. **Enter medical text** in English to translate
            2. **Select translation model** (English → French/Spanish/German)
            3. **Provide reference translation(s)** for BLEU evaluation
            4. **Click 'Translate & Evaluate'** to get translation and BLEU score
            5. **Use 'Multiple Translation Evaluation'** to compare different translations

            ### 🏥 Healthcare Domain Features
            - Medical terminology preprocessing
            - Domain-aware translation
            - Comprehensive BLEU analysis with n-gram precisions
            - Brevity penalty calculation
            - Translation history tracking

            ### 📊 BLEU Score Interpretation
            - **90-100**: Excellent (near-perfect)
            - **80-89**: Very Good (high quality)
            - **70-79**: Good (minor errors)
            - **60-69**: Fair (some errors)
            - **50-59**: Acceptable (basic meaning preserved)
            - **< 50**: Poor (many errors)
            """)

        return app

    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        app = self.create_interface()

        # Merge with config settings
        launch_kwargs = {
            "server_port": self.config['ui']['port'],
            "share": self.config['ui']['share'],
            "debug": self.config['ui']['debug']
        }
        launch_kwargs.update(kwargs)

        print(f"🚀 Launching Healthcare NMT Application on port {launch_kwargs['server_port']}")
        print(f"📡 Shareable link: {'Yes' if launch_kwargs['share'] else 'No'}")

        return app.launch(**launch_kwargs)