"""Flask web application for the Smart Research Assistant prototype."""

import hashlib
import logging
import os
import secrets
import tempfile
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from src.config import get_settings
from src.ingestion import delete_namespace, ingest_file
from src.summarizer import answer_query

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_FILES_PER_UPLOAD = 5
MAX_QUESTION_LENGTH = 2000
SESSION_TTL_SECONDS = 12 * 60 * 60
COLLABORATION_URL = (
    "https://github.com/HarshavardhanKakarla/Research-Buddy/issues/new"
    "?template=collaboration.yml"
)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES + 1024 * 1024,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "false").lower() == "true",
    )
    settings = get_settings()

    # API keys and document manifests stay in process memory, never in the
    # browser cookie or on the app's local filesystem.
    session_data = {}

    @app.before_request
    def bind_session():
        now = time.time()
        expired = [
            session_id
            for session_id, state in session_data.items()
            if now - state["last_seen"] > SESSION_TTL_SECONDS
        ]
        for session_id in expired:
            state = session_data[session_id]
            if not state["manifest"]:
                del session_data[session_id]
                continue
            try:
                delete_namespace(settings, state["namespace"])
            except Exception:
                logger.exception("Could not clean up expired Pinecone session namespace")
                state["last_seen"] = now
            else:
                del session_data[session_id]

        session_id = session.get("session_id")
        if not session_id:
            session_id = secrets.token_urlsafe(32)
            session["session_id"] = session_id
        session.permanent = True
        session.setdefault("csrf_token", secrets.token_urlsafe(32))

        namespace = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
        state = session_data.setdefault(
            session_id,
            {"api_key": None, "manifest": {}, "namespace": namespace},
        )
        state["last_seen"] = now
        g.session_state = state
        g.namespace = namespace

    @app.after_request
    def set_security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; form-action 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
        )
        if app.config["SESSION_COOKIE_SECURE"]:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    def validate_csrf():
        submitted = request.form.get("csrf_token", "")
        expected = session.get("csrf_token", "")
        if not submitted or not secrets.compare_digest(submitted, expected):
            abort(400, description="The form expired. Reload the page and try again.")

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            api_key_set=bool(g.session_state["api_key"]),
            documents=g.session_state["manifest"],
            result=None,
            collaboration_url=COLLABORATION_URL,
            csrf_token=session["csrf_token"],
        )

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.post("/api-key")
    def set_api_key():
        validate_csrf()
        api_key = request.form.get("api_key", "").strip()
        if not api_key or len(api_key) > 500 or any(ord(char) < 32 for char in api_key):
            flash("Enter a valid OpenAI API key.", "error")
        else:
            g.session_state["api_key"] = api_key
            flash(
                "Your key is set for this browser session. It is not stored in your browser cookie.",
                "success",
            )
        return redirect(url_for("index"))

    @app.post("/api-key/clear")
    def clear_api_key():
        validate_csrf()
        g.session_state["api_key"] = None
        flash("Your API key has been cleared from this session.", "success")
        return redirect(url_for("index"))

    @app.post("/upload")
    def upload():
        validate_csrf()
        if not g.session_state["api_key"]:
            flash("Add your OpenAI API key for this session before uploading documents.", "error")
            return redirect(url_for("index"))

        files = [item for item in request.files.getlist("documents") if item.filename]
        if not files:
            flash("Choose at least one PDF, TXT, or Markdown file.", "error")
            return redirect(url_for("index"))
        if len(files) > MAX_FILES_PER_UPLOAD:
            flash(f"Upload no more than {MAX_FILES_PER_UPLOAD} files at once.", "error")
            return redirect(url_for("index"))

        session_settings = replace(settings, openai_api_key=g.session_state["api_key"])
        for uploaded in files:
            filename = secure_filename(uploaded.filename or "")
            extension = Path(filename).suffix.lower()
            if not filename or extension not in ALLOWED_EXTENSIONS:
                flash("Only PDF, TXT, and Markdown files are supported.", "error")
                continue

            try:
                content = uploaded.stream.read(MAX_UPLOAD_BYTES + 1)
                if len(content) > MAX_UPLOAD_BYTES:
                    flash(f"{filename} exceeds the 16 MB upload limit.", "error")
                    continue
                with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name
                try:
                    outcome = ingest_file(
                        temp_path,
                        session_settings,
                        namespace=g.namespace,
                        manifest=g.session_state["manifest"],
                        source_name=filename,
                    )
                finally:
                    os.unlink(temp_path)
                flash(
                    f"{filename}: {outcome['status']} ({outcome['chunks']} chunks).",
                    "success",
                )
            except Exception:
                logger.exception("Document ingestion failed for %s", filename)
                flash(
                    f"Could not ingest {filename}. Check the server logs and API configuration.",
                    "error",
                )
        return redirect(url_for("index"))

    @app.post("/ask")
    def ask():
        validate_csrf()
        api_key = g.session_state["api_key"]
        if not api_key:
            flash("Add your OpenAI API key for this session before asking a question.", "error")
            return redirect(url_for("index"))

        question = request.form.get("question", "").strip()
        mode = request.form.get("mode", "auto")
        section = request.form.get("section", "any")
        source = request.form.get("source", "all")
        valid_modes = {"auto", "bullets", "comparison", "pros_cons"}
        valid_sections = {
            "any", "abstract", "introduction", "methods", "results",
            "discussion", "comparison", "conclusion",
        }
        documents = g.session_state["manifest"]
        if not question or len(question) > MAX_QUESTION_LENGTH:
            flash("Enter a question of 1–2,000 characters.", "error")
            return redirect(url_for("index"))
        if mode not in valid_modes or section not in valid_sections:
            abort(400, description="Invalid answer style or section.")
        if source != "all" and source not in documents:
            abort(400, description="That document is not available in this session.")
        if not documents:
            flash("Upload at least one document before asking a question.", "error")
            return redirect(url_for("index"))

        try:
            session_settings = replace(settings, openai_api_key=api_key)
            result = answer_query(
                question,
                session_settings,
                mode=mode,
                source=None if source == "all" else source,
                section=None if section == "any" else section,
                namespace=g.namespace,
            )
        except Exception:
            logger.exception("Research query failed")
            flash("The query could not be completed. Check the server logs and API configuration.", "error")
            return redirect(url_for("index"))

        return render_template(
            "index.html",
            api_key_set=True,
            documents=documents,
            result=result.model_dump(),
            collaboration_url=COLLABORATION_URL,
            csrf_token=session["csrf_token"],
            selected_mode=mode,
            selected_section=section,
            selected_source=source,
            previous_question=question,
        )

    @app.errorhandler(413)
    def upload_too_large(_error):
        flash("The request is too large. Each uploaded file must be 16 MB or smaller.", "error")
        return redirect(url_for("index")), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run()
