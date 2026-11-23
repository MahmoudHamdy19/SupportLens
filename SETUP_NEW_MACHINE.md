# Running SupportLens on a new machine

1. Install Python 3.10 or newer (python.org; tick "Add Python to PATH").
2. Unzip the project anywhere, then double-click `setup.bat` (or run it in a terminal). It creates a
   virtual environment in `.venv`, installs `requirements.txt`, creates `.env` from `.env.example` and
   runs the offline test suite.
3. Open `.env` in a text editor and set `GITHUB_TOKEN=` to a GitHub personal access token
   (github.com → Settings → Developer settings → Personal access tokens; read-only public access is enough).
   Without a token GitHub allows only 60 requests per hour and the collection becomes very slow.
4. Continue the data collection (resumable; it continues from the records already in `data/`):
   ```
   .venv\Scripts\python scripts\collect_dataset.py --target 300
   ```
   or double-click `collect.bat`. If it stops (network, sleep, closed window), run it again and it
   continues where it stopped. It ends by itself at 300 policies.
5. When collection is finished, follow the workflow in `README.md` (sampling, annotation, adjudication,
   splits, GPU training on Colab, evaluation).

If you use an AI assistant (Gemini CLI, Claude Code, others) on this project, it must follow
`AGENTS.md`. That file is also provided as `GEMINI.md` and `CLAUDE.md`, so the assistant loads it
automatically.
