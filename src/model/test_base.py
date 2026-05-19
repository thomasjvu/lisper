#!/usr/bin/env python3
"""Smoke-test Gemma 4 E2B inference with Unsloth."""

from __future__ import annotations


def test_gemma_inference() -> bool:
    """Load Gemma 4 E2B and run a small text-only check."""

    print("=" * 50)
    print("Testing Gemma 4 E2B Inference")
    print("=" * 50)

    try:
        import torch
        from unsloth import FastVisionModel

        print("\n[1/4] Loading Gemma 4 E2B model...")
        model, processor = FastVisionModel.from_pretrained(
            model_name="google/gemma-4-E2B-it",
            max_seq_length=2048,
            load_in_4bit=True,
            full_finetuning=False,
        )
        FastVisionModel.for_inference(model)
        print("✓ Model loaded successfully")

        print("\n[2/4] Preparing prompt...")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a helpful speech therapy assistant.\n\n"
                            "A learner said 'sun' and it sounded a bit like 'thun'. "
                            "Give concise, encouraging corrective feedback."
                        ),
                    }
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        print("✓ Prompt tokenized")

        print("\n[3/4] Generating response...")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
        )

        prompt_length = inputs["input_ids"].shape[1]
        response = processor.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
        print("✓ Generation complete!")

        print("\n[4/4] Results:")
        print("-" * 50)
        print("Input: User said 'sun' (sounds like 'thun')")
        print(f"Output: {response}")
        return True

    except ImportError as error:
        print(f"\n✗ Missing dependency: {error}")
        print("\nTo install Unsloth:")
        print("  pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo")
        return False
    except Exception as error:
        print(f"\n✗ Error: {error}")
        return False


def check_environment() -> bool:
    """Check whether the local environment is ready."""

    print("\n" + "=" * 50)
    print("Environment Check")
    print("=" * 50)

    import torch

    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    try:
        import transformers

        print(f"✓ Transformers: {transformers.__version__}")
    except Exception:
        print("✗ Transformers not installed")

    return True


if __name__ == "__main__":
    check_environment()
    print("\n")
    test_gemma_inference()
