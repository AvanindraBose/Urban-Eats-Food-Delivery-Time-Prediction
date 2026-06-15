CONTRIBUTING
============

Minimal instructions for local development and contributing.

1. Setup (recommended: use virtualenv or Poetry/uv)

   - Create and activate a virtual environment (example using venv):

     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```

   - If you use `uv` (recommended here), install and sync dependencies:

     ```bash
     pip install uv
     uv sync --extra test --extra pipeline
     ```

2. Run the API locally

   - Start the FastAPI app with Uvicorn:

     ```bash
     uvicorn backend.main:app --reload
     ```

   - The API will be available at http://127.0.0.1:8000 by default.

3. Run tests

   - Run unit tests for the API:

     ```bash
     uv run pytest tests/api/unit -q
     ```

   - Run model tests:

     ```bash
     uv run pytest tests/model -q
     ```

4. Code style and linting

   - Run linters and formatters configured in the repo (e.g., `ruff`, `black`) before committing.

5. Making a contribution

   - Create a feature branch: `git checkout -b feat/my-change`
   - Keep changes small and focused, add tests when applicable.
   - Open a pull request with a clear description and link to any related issues.

6. Secrets and CI

   - Do not commit secrets. Configure required secrets in the repository settings for CI.

If you want, I can add a short PR template and a GitHub Actions job that runs linters on PRs.
