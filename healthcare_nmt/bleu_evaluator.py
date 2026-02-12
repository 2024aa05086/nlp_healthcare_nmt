"""
BLEU Score Evaluation System for Machine Translation
"""

from sacrebleu import sentence_bleu, corpus_bleu
import numpy as np
from typing import List, Dict, Optional, Union
from collections import Counter
import pandas as pd


class BLEUEvaluator:
    """Comprehensive BLEU Score Evaluation System"""

    def __init__(self, config: Dict = None):
        """
        Initialize BLEU Evaluator

        Args:
            config: Configuration dictionary
        """
        if config is None:
            from .utils import load_config
            config = load_config()

        self.config = config
        self.bleu_config = config['evaluation']['bleu']
        self.ngram_config = config['evaluation']['ngram']

    def calculate_bleu(self,
                       candidate: str,
                       references: List[str],
                       detailed: bool = True) -> Dict:
        """
        Calculate BLEU score with detailed breakdown

        Args:
            candidate: Machine translation output
            references: List of reference translations
            detailed: Whether to return detailed analysis

        Returns:
            Dictionary with BLEU score and components
        """
        # Validate inputs
        if not candidate or not references:
            return self._get_empty_result()

        # Calculate BLEU using sacrebleu
        bleu_result = sentence_bleu(
            candidate,
            references,
            tokenize=self.bleu_config['tokenize'],
            smooth_method=self.bleu_config['smooth_method']
        )

        if not detailed:
            return {"bleu_score": bleu_result.score}

        # Get detailed analysis
        detailed_result = {
            "bleu_score": bleu_result.score,
            "score_details": {
                "score": bleu_result.score,
                "counts": bleu_result.counts,
                "totals": bleu_result.totals,
                "precisions": bleu_result.precisions,
                "bp": bleu_result.bp,
                "sys_len": bleu_result.sys_len,
                "ref_len": bleu_result.ref_len,
            },
            "brevity_penalty": bleu_result.bp,
            "sentence_lengths": {
                "candidate": len(candidate.split()),
                "references": [len(ref.split()) for ref in references]
            }
        }

        # Add n-gram precisions
        ngram_precisions = self._calculate_detailed_ngram_precisions(
            candidate, references
        )
        detailed_result["ngram_precisions"] = ngram_precisions

        # Add interpretation
        detailed_result["interpretation"] = self._interpret_bleu_score(
            bleu_result.score
        )

        return detailed_result

    def calculate_corpus_bleu(self,
                              candidates: List[str],
                              references_list: List[List[str]]) -> Dict:
        """
        Calculate corpus-level BLEU score

        Args:
            candidates: List of candidate translations
            references_list: List of reference translations for each candidate

        Returns:
            Corpus BLEU results
        """
        # Flatten references for sacrebleu
        if all(isinstance(refs, list) for refs in references_list):
            # Ensure all candidates have same number of references
            max_refs = max(len(refs) for refs in references_list)
            references = []
            for refs in references_list:
                refs_extended = refs + [refs[-1]] * (max_refs - len(refs))
                references.append(refs_extended)
            references = list(zip(*references))
        else:
            references = [references_list]

        # Calculate corpus BLEU
        bleu_result = corpus_bleu(
            candidates,
            references,
            tokenize=self.bleu_config['tokenize'],
            smooth_method=self.bleu_config['smooth_method']
        )

        return {
            "corpus_bleu": bleu_result.score,
            "score_details": {
                "score": bleu_result.score,
                "counts": bleu_result.counts,
                "totals": bleu_result.totals,
                "precisions": bleu_result.precisions,
                "bp": bleu_result.bp,
                "sys_len": bleu_result.sys_len,
                "ref_len": bleu_result.ref_len,
            },
            "num_sentences": len(candidates)
        }

    def _calculate_detailed_ngram_precisions(self,
                                             candidate: str,
                                             references: List[str]) -> Dict:
        """Calculate detailed n-gram precision analysis"""
        cand_tokens = candidate.lower().split()
        refs_tokens = [ref.lower().split() for ref in references]

        results = {}

        for n in range(1, self.ngram_config['max_n'] + 1):
            # Get candidate n-grams
            cand_ngrams = self._get_ngrams(cand_tokens, n)
            cand_counts = Counter(cand_ngrams)

            # Get max reference counts
            max_ref_counts = {}
            for ref_tokens in refs_tokens:
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                ref_counts = Counter(ref_ngrams)
                for ngram, count in ref_counts.items():
                    max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)

            # Calculate statistics
            total_ngrams = sum(cand_counts.values())
            clipped_matches = sum(
                min(count, max_ref_counts.get(ngram, 0))
                for ngram, count in cand_counts.items()
            )

            precision = (clipped_matches / total_ngrams * 100) if total_ngrams > 0 else 0

            results[n] = {
                "precision": round(precision, 2),
                "total_ngrams": total_ngrams,
                "clipped_matches": clipped_matches,
                "unique_ngrams": len(cand_counts),
                "ngram_examples": list(cand_counts.keys())[:5]  # First 5 examples
            }

        return results

    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Extract n-grams from tokens"""
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def _interpret_bleu_score(self, score: float) -> Dict:
        """Interpret BLEU score quality"""
        if score >= 90:
            quality = "Excellent"
            description = "Near-perfect translation"
        elif score >= 80:
            quality = "Very Good"
            description = "High quality translation"
        elif score >= 70:
            quality = "Good"
            description = "Good quality with minor errors"
        elif score >= 60:
            quality = "Fair"
            description = "Acceptable with some errors"
        elif score >= 50:
            quality = "Acceptable"
            description = "Basic meaning preserved"
        elif score >= 40:
            quality = "Poor"
            description = "Many errors, meaning partially preserved"
        else:
            quality = "Very Poor"
            description = "Severe translation errors"

        return {
            "quality": quality,
            "description": description,
            "score_range": self._get_score_range(score)
        }

    def _get_score_range(self, score: float) -> str:
        """Get score range category"""
        ranges = [
            (90, 100, "90-100"),
            (80, 90, "80-89"),
            (70, 80, "70-79"),
            (60, 70, "60-69"),
            (50, 60, "50-59"),
            (40, 50, "40-49"),
            (0, 40, "0-39")
        ]

        for lower, upper, label in ranges:
            if lower <= score < upper:
                return label
        return "0-39"

    def _get_empty_result(self) -> Dict:
        """Return empty result for invalid inputs"""
        return {
            "bleu_score": 0.0,
            "brevity_penalty": 1.0,
            "ngram_precisions": {n: {"precision": 0.0} for n in range(1, 5)},
            "interpretation": {
                "quality": "Invalid",
                "description": "No candidate or references provided",
                "score_range": "N/A"
            }
        }

    def evaluate_multiple(self,
                          candidates: List[str],
                          references_list: List[List[str]]) -> pd.DataFrame:
        """
        Evaluate multiple candidate translations

        Args:
            candidates: List of candidate translations
            references_list: List of reference lists

        Returns:
            DataFrame with evaluation results
        """
        results = []

        for i, (cand, refs) in enumerate(zip(candidates, references_list)):
            eval_result = self.calculate_bleu(cand, refs, detailed=True)

            results.append({
                "ID": i + 1,
                "Candidate": cand[:100] + "..." if len(cand) > 100 else cand,
                "BLEU Score": round(eval_result["bleu_score"], 2),
                "Quality": eval_result["interpretation"]["quality"],
                "1-gram Precision": eval_result["ngram_precisions"][1]["precision"],
                "2-gram Precision": eval_result["ngram_precisions"][2]["precision"],
                "3-gram Precision": eval_result["ngram_precisions"][3]["precision"],
                "4-gram Precision": eval_result["ngram_precisions"][4]["precision"],
                "Brevity Penalty": round(eval_result["brevity_penalty"], 3),
                "Candidate Length": len(cand.split()),
                "Avg Reference Length": np.mean([len(r.split()) for r in refs])
            })

        return pd.DataFrame(results)

    def generate_report(self,
                        candidates: List[str],
                        references_list: List[List[str]]) -> Dict:
        """
        Generate comprehensive evaluation report

        Args:
            candidates: List of candidate translations
            references_list: List of reference lists

        Returns:
            Comprehensive report dictionary
        """
        # Individual evaluations
        individual_results = []
        for cand, refs in zip(candidates, references_list):
            result = self.calculate_bleu(cand, refs, detailed=True)
            individual_results.append(result)

        # Corpus-level evaluation
        corpus_result = self.calculate_corpus_bleu(candidates, references_list)

        # Statistics
        bleu_scores = [r["bleu_score"] for r in individual_results]

        report = {
            "summary": {
                "total_sentences": len(candidates),
                "average_bleu": np.mean(bleu_scores),
                "std_bleu": np.std(bleu_scores),
                "min_bleu": np.min(bleu_scores),
                "max_bleu": np.max(bleu_scores),
                "corpus_bleu": corpus_result["corpus_bleu"]
            },
            "distribution": {
                "excellent": sum(1 for s in bleu_scores if s >= 90),
                "very_good": sum(1 for s in bleu_scores if 80 <= s < 90),
                "good": sum(1 for s in bleu_scores if 70 <= s < 80),
                "fair": sum(1 for s in bleu_scores if 60 <= s < 70),
                "acceptable": sum(1 for s in bleu_scores if 50 <= s < 60),
                "poor": sum(1 for s in bleu_scores if s < 50)
            },
            "individual_results": individual_results,
            "corpus_result": corpus_result
        }

        return report