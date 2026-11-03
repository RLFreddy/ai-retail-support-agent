# Deploying to a Hugging Face Space

The Space builds the repository's `Dockerfile`, whose default command serves
the API and the web UI on port 7860. Only the Space front matter differs, so
the Space's `README.md` is `deploy/huggingface/README.md`.

1. Create a Space with the **Docker** SDK (blank template).
2. In *Settings → Variables and secrets*, add the secret `GROQ_API_KEY`
   (optional: without it the Space runs the offline mock model).
3. Push the code with the Space README:

   ```bash
   git clone https://huggingface.co/spaces/<user>/<space> /tmp/space
   git archive HEAD | tar -x -C /tmp/space
   cp deploy/huggingface/README.md /tmp/space/README.md
   cd /tmp/space && git add -A && git commit -m "deploy: sync from GitHub" && git push
   ```
