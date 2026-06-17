# Deploy the live demo on Hugging Face Spaces (free)

The offline demo ([`app.py`](../app.py)) runs the **real** retrieval pipeline with
deterministic stand-in models — **no API keys, no GPU, no database**. That makes it
ideal for a free, always-on Hugging Face Space.

## One-time setup (~3 minutes)

1. Go to **https://huggingface.co/new-space**.
2. **Owner**: you · **Space name**: `clipopedia` · **SDK**: **Gradio** · Hardware: **CPU basic (free)** · Visibility: **Public**.
3. In the new Space, add two files (via the web "Files" tab or `git push`):

   **`app.py`** — copy this repo's [`app.py`](../app.py) verbatim.

   **`requirements.txt`**:
   ```text
   git+https://github.com/akira231097/clipopedia.git
   gradio
   ```

4. The Space builds automatically and serves the demo. First build takes ~1–2 minutes.

That's it — the Space installs the package straight from GitHub, then runs `app.py`.

## After it's live

Add the Space URL to the top of the main [README](../README.md), e.g.:

```markdown
🔴 **Live demo:** https://huggingface.co/spaces/<you>/clipopedia
```

## Notes

- The demo intentionally uses the **offline** backend (`CLIPOPEDIA_BACKEND=demo`,
  the default), so there are no secrets to configure.
- To run the same app locally: `pip install -e . gradio && python app.py`.
