"""
Persistent local model server.

Keeps the MLX model loaded in memory between requests so callers
skip the ~20s load time. Start once and leave running:

    python model_server.py

summarizer_local.py auto-detects this server and routes through it.
If the server isn't running, it falls back to loading the model directly.
"""

from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8321
MODEL_ID = "mlx-community/Qwen3.5-9B-MLX-4bit"
REPETITION_WINDOW = 600


def _load_model():
    from mlx_lm import load

    print(f"Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)
    print("Model loaded.")
    return model, tokenizer


def _generate(model, tokenizer, system_prompt: str, user_content: str, max_tokens: int) -> str:
    from mlx_lm import stream_generate

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if prompt.rstrip().endswith("<think>"):
        prompt = prompt.rstrip() + "\n</think>\n\n"

    text = ""
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        text += response.text
        if len(text) > REPETITION_WINDOW * 2:
            tail = text[-REPETITION_WINDOW:]
            if tail in text[:-REPETITION_WINDOW]:
                print(f"[server] Repetition loop at {len(text)} chars, stopping")
                break

    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


class Handler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None

    def do_POST(self):
        if self.path != "/generate":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        result = _generate(
            self.model,
            self.tokenizer,
            body["system_prompt"],
            body["user_content"],
            body.get("max_tokens", 8192),
        )

        response = json.dumps({"content": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        # Only log errors, not every request
        if args and "200" not in str(args[0]):
            super().log_message(format, *args)


if __name__ == "__main__":
    model, tokenizer = _load_model()
    Handler.model = model
    Handler.tokenizer = tokenizer

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Serving on http://127.0.0.1:{PORT}")
    print("Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
