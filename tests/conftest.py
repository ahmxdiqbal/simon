"""Shared test setup. Provides a dummy API key so the DeepSeek module imports
without real credentials; the network client is mocked in tests that need it."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-dummy-key")
