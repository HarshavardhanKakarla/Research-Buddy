import io
import os
import re
import unittest
from unittest.mock import patch

os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("FLASK_SECRET_KEY", "test-flask-secret-key")
os.environ["COOKIE_SECURE"] = "false"

import app as web_app


class ResearchBuddyWebTests(unittest.TestCase):
    def setUp(self):
        web_app.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
        self.client = web_app.app.test_client()

    def csrf_token(self, client=None):
        response = (client or self.client).get("/")
        match = re.search(
            rb'name="csrf_token" value="([^"]+)"',
            response.data,
        )
        self.assertIsNotNone(match)
        return match.group(1).decode()

    def set_key(self, client=None):
        client = client or self.client
        return client.post(
            "/api-key",
            data={
                "csrf_token": self.csrf_token(client),
                "api_key": "sk-test-session-key-do-not-store",
            },
            follow_redirects=True,
        )

    def test_home_displays_demo_notice_and_collaboration_link(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prototype demo", response.data)
        self.assertIn(b"issues/new?template=collaboration.yml", response.data)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])

    def test_health_check(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"status": "ok"})

    def test_api_key_is_not_saved_in_signed_browser_cookie(self):
        self.set_key()
        cookie = self.client.get_cookie("session")
        self.assertIsNotNone(cookie)
        serializer = web_app.app.session_interface.get_signing_serializer(web_app.app)
        cookie_values = serializer.loads(cookie.value)
        self.assertNotIn("api_key", cookie_values)
        self.assertNotIn("sk-test-session-key-do-not-store", cookie.value)
        self.assertIn(b"Key active for this session", self.client.get("/").data)

    def test_csrf_is_required_for_key_submission(self):
        response = self.client.post("/api-key", data={"api_key": "sk-test-session-key"})
        self.assertEqual(response.status_code, 400)

    @patch.object(web_app, "ingest_file")
    def test_upload_uses_session_namespace_and_removes_temporary_file(self, ingest):
        self.set_key()
        ingest.return_value = {"source": "paper.txt", "status": "ingested", "chunks": 1}
        token = self.csrf_token()
        response = self.client.post(
            "/upload",
            data={
                "csrf_token": token,
                "documents": (io.BytesIO(b"research text"), "paper.txt"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        ingest.assert_called_once()
        args, kwargs = ingest.call_args
        self.assertFalse(os.path.exists(args[0]))
        self.assertEqual(kwargs["source_name"], "paper.txt")
        self.assertTrue(kwargs["namespace"])
        self.assertIn(b"paper.txt", response.data)

    @patch.object(web_app, "answer_query")
    @patch.object(web_app, "ingest_file")
    def test_sessions_have_separate_namespaces_and_queries(self, ingest, answer):
        def fake_ingest(_path, _settings, namespace, manifest, source_name):
            manifest[source_name] = {"hash": "test-hash", "chunks": 1, "sections": []}
            return {"source": source_name, "status": "ingested", "chunks": 1}

        ingest.side_effect = fake_ingest
        answer.return_value.model_dump.return_value = {
            "summary": "A grounded summary.",
            "bullets": None,
            "comparison": None,
            "pros_cons": None,
            "sources": ["paper.txt"],
        }

        for client in (self.client, web_app.app.test_client()):
            self.set_key(client)
            client.post(
                "/upload",
                data={
                    "csrf_token": self.csrf_token(client),
                    "documents": (io.BytesIO(b"research text"), "paper.txt"),
                },
                content_type="multipart/form-data",
            )
        namespaces = [call.kwargs["namespace"] for call in ingest.call_args_list]
        self.assertEqual(len(namespaces), 2)
        self.assertNotEqual(namespaces[0], namespaces[1])

        token = self.csrf_token()
        response = self.client.post(
            "/ask",
            data={
                "csrf_token": token,
                "question": "What is the finding?",
                "mode": "bullets",
                "section": "any",
                "source": "all",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"A grounded summary.", response.data)
        self.assertEqual(answer.call_args.kwargs["namespace"], namespaces[0])


if __name__ == "__main__":
    unittest.main()
