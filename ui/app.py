import json
import os

import gradio as gr
import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120


# ── Auth ──────────────────────────────────────────────────────────────────────

def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def register(name: str, password: str):
    if not name.strip() or not password:
        return None, "⚠️ Name and password are required."
    try:
        r = httpx.post(
            f"{API_URL}/api/v1/auth/register",
            json={"name": name.strip(), "password": password},
            timeout=10,
        )
        if r.status_code == 201:
            return r.json()["access_token"], f"✅ Registered as **{name.strip()}**"
        if r.status_code == 409:
            return None, "⚠️ Name already taken — try logging in instead."
        if r.status_code == 422:
            return None, "⚠️ Password must be at least 8 characters."
        return None, f"⚠️ Registration failed ({r.status_code})"
    except httpx.ConnectError:
        return None, "⚠️ Cannot reach the API — is the server running?"


def login(name: str, password: str):
    if not name.strip() or not password:
        return None, "⚠️ Name and password are required."
    try:
        r = httpx.post(
            f"{API_URL}/api/v1/auth/token",
            json={"name": name.strip(), "password": password},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()["access_token"], f"✅ Logged in as **{name.strip()}**"
        if r.status_code == 401:
            return None, "⚠️ Invalid credentials."
        return None, f"⚠️ Login failed ({r.status_code})"
    except httpx.ConnectError:
        return None, "⚠️ Cannot reach the API — is the server running?"


def logout(token: str | None):
    if not token:
        return None, "⚠️ Not logged in."
    try:
        httpx.post(f"{API_URL}/api/v1/auth/logout", headers=_auth_headers(token), timeout=10)
    except Exception:
        pass
    return None, "✅ Logged out."


# ── Documents ─────────────────────────────────────────────────────────────────

def _load_documents(token: str | None) -> tuple[str, dict, list]:
    """Returns (markdown table, {filename: doc_id} dict, dropdown choices)."""
    if not token:
        return "_Authenticate to see your documents._", {}, []
    try:
        r = httpx.get(f"{API_URL}/api/v1/documents/", headers=_auth_headers(token), timeout=10)
        if r.status_code != 200:
            return f"⚠️ Could not load documents ({r.status_code})", {}, []
        data = r.json()
        if data["total"] == 0:
            return "_No documents uploaded yet._", {}, []
        doc_dict = {doc["filename"]: doc["document_id"] for doc in data["results"]}
        rows = ["| Filename | Status | Chunks |", "|---|---|---|"]
        for doc in data["results"]:
            icon = "✅ ready" if doc["status"] == "ready" else "⏳ processing"
            rows.append(f"| {doc['filename']} | {icon} | {doc['chunk_count']} |")
        return "\n".join(rows), doc_dict, list(doc_dict.keys())
    except httpx.ConnectError:
        return "⚠️ Cannot reach the API.", {}, []


def upload_documents(files, token: str | None):
    if not token:
        return "⚠️ Please authenticate first.", *_load_documents(token)
    if not files:
        return "⚠️ No files selected.", *_load_documents(token)

    files = files if isinstance(files, list) else [files]
    results = []
    for file in files:
        file_path = file.name if hasattr(file, "name") else str(file)
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                r = httpx.post(
                    f"{API_URL}/api/v1/documents/upload",
                    files={"file": (filename, f, "application/octet-stream")},
                    headers=_auth_headers(token),
                    timeout=60,
                )
            if r.status_code == 202:
                results.append(f"✅ **{filename}** uploaded")
            elif r.status_code == 200:
                results.append(f"ℹ️ **{filename}** already exists")
            elif r.status_code == 400:
                results.append(f"⚠️ **{filename}**: {r.json().get('detail', 'Bad request')}")
            elif r.status_code == 403:
                return "⚠️ Write scope required to upload.", *_load_documents(token)
            else:
                results.append(f"⚠️ **{filename}** failed ({r.status_code})")
        except httpx.ConnectError:
            results.append(f"⚠️ **{filename}**: cannot reach API")

    status = "\n".join(results)
    return status, *_load_documents(token)


def delete_document(filename: str | None, docs_state: dict, token: str | None):
    if not token:
        return "⚠️ Please authenticate first.", *_load_documents(token)
    if not filename:
        return "⚠️ Select a document to delete.", *_load_documents(token)
    doc_id = docs_state.get(filename)
    if not doc_id:
        return "⚠️ Document not found in list — try refreshing.", *_load_documents(token)
    try:
        r = httpx.delete(
            f"{API_URL}/api/v1/documents/{doc_id}",
            headers=_auth_headers(token),
            timeout=10,
        )
        if r.status_code == 204:
            status = f"🗑️ **{filename}** deleted."
        elif r.status_code == 403:
            status = "⚠️ Write scope required to delete."
        elif r.status_code == 404:
            status = f"⚠️ **{filename}** not found."
        else:
            status = f"⚠️ Delete failed ({r.status_code})"
    except httpx.ConnectError:
        status = "⚠️ Cannot reach the API."
    return status, *_load_documents(token)


# ── Models ────────────────────────────────────────────────────────────────────

def fetch_models() -> tuple[list[str], str]:
    try:
        r = httpx.get(f"{API_URL}/api/v1/models", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data["models"], data["default"]
    except Exception:
        pass
    return ["llama3.2:3b"], "llama3.2:3b"


# ── Chat ──────────────────────────────────────────────────────────────────────

def chat(message: str, history: list, token: str | None, session_id: str | None, model: str | None):
    if not message.strip():
        yield history, session_id
        return

    if not token:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please authenticate first."},
        ]
        yield history, session_id
        return

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield history, session_id

    payload: dict = {"message": message, "top_k": 5}
    if session_id:
        payload["session_id"] = session_id
    if model:
        payload["model"] = model

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream(
                "POST",
                f"{API_URL}/api/v1/chat",
                json=payload,
                headers=_auth_headers(token),
            ) as response:
                response.raise_for_status()
                sources: list[dict] = []
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        if sources:
                            seen: set[str] = set()
                            unique_sources = []
                            for s in sources:
                                if s["filename"] not in seen:
                                    seen.add(s["filename"])
                                    unique_sources.append(s)

                            filenames = ", ".join(s["filename"] for s in unique_sources)
                            footer_parts = [f"📄 **Sources:** {filenames}"]

                            all_entities: dict[str, set] = {}
                            for s in unique_sources:
                                for etype, values in s.get("entities", {}).items():
                                    all_entities.setdefault(etype, set()).update(values)

                            if all_entities:
                                entity_str = " | ".join(
                                    f"{k}: {', '.join(sorted(v))}"
                                    for k, v in sorted(all_entities.items())
                                )
                                footer_parts.append(f"🏷️ **Entities:** {entity_str}")

                            history[-1]["content"] += "\n\n---\n" + "\n\n".join(footer_parts)
                            yield history, session_id
                        break
                    try:
                        event = json.loads(data)
                        if event["type"] == "session" and session_id is None:
                            session_id = event["session_id"]
                        elif event["type"] == "sources":
                            sources = event.get("sources", [])
                        elif event["type"] == "token":
                            history[-1]["content"] += event["token"]
                            yield history, session_id
                        elif event["type"] == "error":
                            history[-1]["content"] = f"⚠️ {event['detail']}"
                            yield history, session_id
                    except json.JSONDecodeError:
                        pass
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            history[-1]["content"] = "⚠️ Session expired — please log in again."
        elif e.response.status_code == 404:
            history[-1]["content"] = "⚠️ No documents found. Upload something first."
        else:
            history[-1]["content"] = f"⚠️ API error ({e.response.status_code})"
        yield history, session_id
    except Exception as e:
        history[-1]["content"] = f"⚠️ Connection error: {e}"
        yield history, session_id


def clear_chat():
    return [], None


# ── Layout ────────────────────────────────────────────────────────────────────

_initial_models, _default_model = fetch_models()

with gr.Blocks(title="RAG Document Q&A") as demo:
    token_state = gr.State(None)
    session_state = gr.State(None)
    docs_state = gr.State({})  # {filename: doc_id}

    gr.Markdown(
        "# RAG Document Q&A\n"
        "Upload PDF or image documents, then ask questions grounded in their content. "
        "Responses stream in real time from a local LLM."
    )

    with gr.Accordion("1 — Authenticate", open=True):
        with gr.Row():
            name_input = gr.Textbox(label="Tenant name", placeholder="my-organisation")
            pass_input = gr.Textbox(label="Password", type="password", placeholder="minimum 8 characters")
        with gr.Row():
            register_btn = gr.Button("Register")
            login_btn = gr.Button("Login", variant="primary")
            logout_btn = gr.Button("Logout", variant="stop")
        auth_status = gr.Markdown("")

    with gr.Accordion("2 — Documents", open=True):
        file_input = gr.File(
            label="Upload documents",
            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".txt"],
            file_count="multiple",
        )
        with gr.Row():
            upload_btn = gr.Button("Upload", variant="primary")
            refresh_btn = gr.Button("Refresh list")
        upload_status = gr.Markdown("")
        doc_list = gr.Markdown("_Authenticate to see your documents._")
        with gr.Row():
            delete_dropdown = gr.Dropdown(label="Select document to delete", choices=[], scale=4)
            delete_btn = gr.Button("Delete", variant="stop", scale=1)
        delete_status = gr.Markdown("")

    with gr.Accordion("3 — Chat", open=True):
        model_dropdown = gr.Dropdown(
            label="Generation model",
            choices=_initial_models,
            value=_default_model,
            scale=1,
        )
        chatbot = gr.Chatbot(height=450, show_label=False)
        with gr.Row():
            msg_input = gr.Textbox(
                label="Question",
                placeholder="Ask anything about your uploaded documents...",
                scale=5,
                lines=1,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("New Chat", size="sm")

    # ── Event wiring ──────────────────────────────────────────────────────────

    def _after_auth(token):
        md, doc_dict, choices = _load_documents(token)
        return md, doc_dict, gr.update(choices=choices)

    register_btn.click(
        fn=register,
        inputs=[name_input, pass_input],
        outputs=[token_state, auth_status],
    ).then(fn=_after_auth, inputs=[token_state], outputs=[doc_list, docs_state, delete_dropdown])

    login_btn.click(
        fn=login,
        inputs=[name_input, pass_input],
        outputs=[token_state, auth_status],
    ).then(fn=_after_auth, inputs=[token_state], outputs=[doc_list, docs_state, delete_dropdown])

    logout_btn.click(
        fn=logout,
        inputs=[token_state],
        outputs=[token_state, auth_status],
    ).then(
        fn=lambda: ("_Authenticate to see your documents._", {}, gr.update(choices=[]), [], None),
        outputs=[doc_list, docs_state, delete_dropdown, chatbot, session_state],
    )

    upload_btn.click(
        fn=upload_documents,
        inputs=[file_input, token_state],
        outputs=[upload_status, doc_list, docs_state, delete_dropdown],
    )

    refresh_btn.click(
        fn=_after_auth,
        inputs=[token_state],
        outputs=[doc_list, docs_state, delete_dropdown],
    )

    delete_btn.click(
        fn=delete_document,
        inputs=[delete_dropdown, docs_state, token_state],
        outputs=[delete_status, doc_list, docs_state, delete_dropdown],
    )

    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot, token_state, session_state, model_dropdown],
        outputs=[chatbot, session_state],
    ).then(fn=lambda: "", outputs=[msg_input])

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot, token_state, session_state, model_dropdown],
        outputs=[chatbot, session_state],
    ).then(fn=lambda: "", outputs=[msg_input])

    clear_btn.click(fn=clear_chat, outputs=[chatbot, session_state])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
