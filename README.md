# Research Buddy

Research Buddy is an open-source prototype for asking focused questions about
research papers. It ingests PDFs and text documents, retrieves relevant
passages with Pinecone, and generates structured answers with source names.

**Repository description:** A source-grounded research assistant prototype for
asking questions about papers, with structured answers and citations.

> **Demo notice:** This is an early prototype, not a finished or confidential
> research service. Do not upload sensitive, private, or unpublished material.

## Architecture

```text
 Browser
   │
   ├── Enter personal OpenAI API key ───────────────┐
   ├── Upload PDF / TXT / Markdown                   │
   └── Ask question                                 │
          │                                         │
          ▼                                         ▼
    Flask app ── per-browser session state ── in-memory key + manifest
          │
          ├── Ingestion: load → split → tag sections → embed
          │                                      │
          │                                      ▼
          │                            Pinecone namespace
          │                            (isolated per session)
          │
          └── Retrieval: embed query → vector search → rerank
                                                    │
                                                    ▼
                                      Context-grounded LLM answer
                                                    │
                                                    ▼
                                  Structured summary + source names
                                                    │
                                                    └── rendered in browser
```

The web layer is in `app.py`. Document loading, chunking, vector storage, and
namespace cleanup are handled by `src/ingestion.py`; semantic retrieval and
reranking live in `src/retrieval.py`; structured answer generation is in
`src/summarizer.py` and its Pydantic models are in `src/schemas.py`.

The browser cookie contains only a signed session identifier and CSRF token;
the submitted OpenAI key and document manifest stay in the app process's
memory. Pinecone stores vectors in a namespace derived from that session ID.
After 12 hours of inactivity, the namespace is deleted on a subsequent request.
This prototype therefore requires a single app process and is not designed for
horizontal scaling or confidential workloads.

## What it does

- Upload PDF, TXT, and Markdown files.
- Ask a question across the current browser session's document library.
- Request bullet summaries, comparisons, or pros and cons, with section filters.
- Use your own OpenAI API key for the current browser session.
- Keep each browser session's vectors in its own Pinecone namespace.

## API usage and privacy

The hosted demo does not provide OpenAI credits. Visitors must supply their own
OpenAI API key and have API billing or available quota enabled; ChatGPT
subscriptions do not automatically include API usage. OpenAI usage is billed
to the visitor's account. The application owner separately provides the
Pinecone service credentials used by the deployment.

The OpenAI key is held in the Flask process's memory for the browser session.
It is not put in the browser cookie or written to disk, and visitors can clear
it from the page. Restarting the app clears in-memory session state. Uploaded
documents are processed through OpenAI embeddings; relevant passages are sent
to OpenAI to generate answers. Vectors are stored in a per-session Pinecone
namespace and deleted after 12 hours of inactivity (when another request
arrives). A hard restart before cleanup may leave vectors behind; remove
orphaned namespaces through Pinecone if needed.
Review the providers' privacy and retention policies before using the demo.

This prototype uses in-memory session state and is intended for a single
application process. Do not run multiple workers/instances without replacing
the in-memory session store with shared server-side storage and configuring
session affinity or a shared session backend. The app does not currently
provide user accounts, authentication, or a production privacy guarantee.

## Run locally

Requirements: Python 3.10+ and a Pinecone account/API key. Each visitor enters
their OpenAI API key in the webpage.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `PINECONE_API_KEY` in `.env`, then run:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Set the generated value as `FLASK_SECRET_KEY` in `.env`. For local HTTP
development set `COOKIE_SECURE=false`; set it to `true` when serving over
HTTPS.

```powershell
flask --app app run
```

Open the local URL printed by Flask. Users enter their OpenAI API key on the
page; there is no need to set `OPENAI_API_KEY` for the web app.

## Deploy

This project includes a `Procfile` for platforms that support Gunicorn:

```text
web: gunicorn app:app
```

Configure these deployment environment variables (never commit secrets):

- `PINECONE_API_KEY` — required; the deployment's Pinecone credential.
- `PINECONE_INDEX_NAME` — optional; defaults to `research-assistant`.
- `FLASK_SECRET_KEY` — required; use a long, random, persistent value.
- `COOKIE_SECURE=true` — enable secure cookies when using HTTPS.

Install dependencies from `requirements.txt` and start the web process using
the `Procfile`. Use a platform with persistent environment configuration and
HTTPS. The current session store requires a single app process; use an external
server-side session store before scaling horizontally. Monitor Pinecone and
OpenAI usage and set provider-side spending limits.

## Batch ingestion

The optional CLI reads `OPENAI_API_KEY` from `.env` or the environment:

```powershell
$env:OPENAI_API_KEY = "your-key"
python scripts/ingest_cli.py --dir .\data\uploads
```

Do not use the CLI key in a public deployment. The web app intentionally asks
each visitor for their own key.

## Collaboration

Want to help build Research Buddy into an end-to-end product? Use the
**Build with me** link on the webpage to open a public collaboration request
on GitHub. Please do not include private information in public issues.

## License

Research Buddy is provided under the MIT License. See [LICENSE](LICENSE).
