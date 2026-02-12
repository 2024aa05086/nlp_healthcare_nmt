#!/usr/bin/env python3
"""
Test script for Healthcare NMT System
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from healthcare_nmt import HealthcareNMT, BLEUEvaluator


def test_translation():
    """Test translation functionality"""
    print("🧪 Testing Healthcare NMT System")
    print("=" * 50)

    # Initialize NMT system
    nmt = HealthcareNMT()

    # Test cases
    test_cases = [
        {
            "text": "Take two tablets after meals for hypertension.",
            "description": "Simple medical instruction"
        },
        {
            "text": "The patient was diagnosed with diabetes and requires insulin treatment.",
            "description": "Diagnosis and treatment"
        },
        {
            "text": "Administer antibiotic 500mg three times daily for seven days.",
            "description": "Prescription instructions"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test['description']}")
        print(f"Source: {test['text']}")

        try:
            translation = nmt.translate(test['text'])
            print(f"Translation: {translation}")
        except Exception as e:
            print(f"❌ Translation failed: {e}")

    print("\n" + "=" * 50)
    print("✅ Translation tests completed")


def test_bleu_evaluation():
    """Test BLEU evaluation functionality"""
    print("\n🧪 Testing BLEU Evaluation System")
    print("=" * 50)

    # Initialize evaluator
    evaluator = BLEUEvaluator()

    # Test case
    candidate = "Le patient doit prendre deux comprimés après les repas."
    references = [
        "Le patient doit prendre deux comprimés après les repas.",
        "Le patient devrait prendre deux comprimés après avoir mangé.",
        "Prenez deux comprimés après les repas, s'il vous plaît."
    ]

    print(f"Candidate: {candidate}")
    print(f"References: {references}")

    # Calculate BLEU
    result = evaluator.calculate_bleu(candidate, references, detailed=True)

    print(f"\n📊 BLEU Score: {result['bleu_score']:.2f}")
    print(f"Brevity Penalty: {result['brevity_penalty']:.3f}")

    print("\nN-gram Precisions:")
    for n in range(1, 5):
        precision = result['ngram_precisions'][n]['precision']
        print(f"  {n}-gram: {precision:.2f}%")

    print(f"\nQuality: {result['interpretation']['quality']}")
    print(f"Description: {result['interpretation']['description']}")

    print("\n" + "=" * 50)
    print("✅ BLEU evaluation tests completed")


def test_multiple_evaluation():
    """Test multiple candidate evaluation"""
    print("\n🧪 Testing Multiple Candidate Evaluation")
    print("=" * 50)

    evaluator = BLEUEvaluator()

    candidates = [
        "Le patient doit prendre des médicaments.",
        "Le patient devrait prendre ses médicaments.",
        "Prenez vos médicaments, s'il vous plaît."
    ]

    references_list = [
        ["Le patient doit prendre ses médicaments."],
        ["Le patient doit prendre ses médicaments."],
        ["Le patient doit prendre ses médicaments."]
    ]

    results = evaluator.evaluate_multiple(candidates, references_list)

    print("Multiple Evaluation Results:")
    print(results.to_string(index=False))

    print("\n" + "=" * 50)
    print("✅ Multiple evaluation tests completed")


def test_model_info():
    """Test model information retrieval"""
    print("\n🧪 Testing Model Information")
    print("=" * 50)

    nmt = HealthcareNMT()
    info = nmt.get_model_info()

    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("✅ Model information tests completed")


def main():
    """Run all tests"""
    print("🚀 Healthcare NMT System - Comprehensive Tests")
    print("=" * 60)

    try:
        test_translation()
        test_bleu_evaluation()
        test_multiple_evaluation()
        test_model_info()

        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
