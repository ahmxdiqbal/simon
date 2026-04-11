import sys
from mlx_lm import load, generate

MODELS = {
    "0.8b": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    "4b":   "mlx-community/Qwen3.5-4B-MLX-4bit",
    "9b":   "mlx-community/Qwen3.5-9B-MLX-4bit",
}

key = sys.argv[1] if len(sys.argv) > 1 else "4b"
if key not in MODELS:
    print(f"Unknown model '{key}'. Choose from: {', '.join(MODELS)}")
    sys.exit(1)

MODEL = MODELS[key]
print(f"Loading {MODEL}...")
model, tokenizer = load(MODEL)
print("Ready. Ctrl+C to exit.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
        break

    if not user_input:
        continue

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_input}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if prompt.rstrip().endswith("<think>"):
        prompt = prompt.rstrip() + "\n</think>\n\n"

    generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=True)
    print()
