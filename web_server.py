"""
ENML Web Server — Browser Chat UI with Full Memory Pipeline.

Provides the same ENML experience as the CLI (fact extraction, memory retrieval,
knowledge graph, authority memory) through a beautiful web interface.

Routes:
    GET  /           → Chat UI
    POST /api/chat   → Send message (SSE streaming response)
    GET  /api/health → Health check
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, Response, jsonify, stream_with_context
from core.orchestrator import Orchestrator
from core.logger import get_logger
from core.config import AI_NAME, AI_HINT, WEB_SERVER_PORT, MAX_REALTIME_INPUT_CHARS
from core.memory.document_ingester import DocumentIngester
from tools.eval_runtime import evaluate as evaluate_runtime_metrics, load_entries as load_runtime_entries
from tools.eval_citations import load as load_citation_entries

logger = get_logger("WebServer")

app = Flask(__name__, template_folder="templates")

# ── Global state ─────────────────────────────────────────────────────────
orchestrator = None
doc_ingester = None
sessions: Dict[str, Dict] = {}  # session_id -> {history: [...], created: datetime}


def get_or_create_session(session_id: str) -> Dict:
    """Get or create a session by ID."""
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "created": datetime.now().isoformat()
        }
    return sessions[session_id]


# ── Input Classification (same logic as CLI) ─────────────────────────────
import re

_DOC_INDICATORS = [
    re.compile(r'^#{1,6}\s+\w', re.MULTILINE),
    re.compile(r'```\w*\s*\n', re.MULTILINE),
    re.compile(r'\|[-:]+\|', re.MULTILINE),
    re.compile(r'[┌┐└┘├┤┬┴┼─│═║╔╗╚╝╠╣╦╩╬]'),
    re.compile(r'^\s*[-*]\s+\[[ x]\]', re.MULTILINE),
]

def classify_input(text: str) -> str:
    """Returns 'conversation' or 'document'."""
    if not text:
        return "conversation"
    text = text.strip()
    if len(text) > MAX_REALTIME_INPUT_CHARS:
        lines = text.split('\n')
        if len(lines) > 5 or len(text) > 1000:
            return "document"
    indicator_hits = sum(1 for p in _DOC_INDICATORS if p.search(text))
    if indicator_hits >= 2:
        return "document"
    lines = text.split('\n')
    if len(lines) > 8 and len(text) / max(len(lines), 1) < 60:
        return "document"
    return "conversation"


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("chat.html", ai_name=AI_NAME)


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "ai_name": AI_NAME,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process a chat message through the full ENML pipeline.
    
    Request JSON: {message: str, session_id: str (optional)}
    Response: SSE stream of content chunks.
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400
    
    user_message = data["message"].strip()
    session_id = data.get("session_id", f"web_{uuid.uuid4().hex[:8]}")
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    session = get_or_create_session(session_id)
    history = session["history"]
    
    input_type = classify_input(user_message)
    
    system_prompt = (
        f"You are {AI_NAME} {AI_HINT}.\n"
        "Keep answers concise and efficient.\n"
    )
    
    def generate():
        nonlocal history
        
        # ── Document ingestion ────────────────────────────────
        if input_type == "document":
            line_count = len(user_message.split('\n'))
            
            # Send ingestion status
            status_msg = json.dumps({
                "type": "status",
                "content": f"📄 Large input detected ({len(user_message)} chars, {line_count} lines) — ingesting as document..."
            })
            yield f"data: {status_msg}\n\n"
            
            try:
                result = doc_ingester.ingest(user_message, source_label="web_pasted_document")
                result_msg = json.dumps({
                    "type": "status",
                    "content": f"✅ Document ingested: {result['sections']} sections, {result.get('summaries_stored', 0)} summaries, {result['facts_extracted']} facts extracted"
                })
                yield f"data: {result_msg}\n\n"
            except Exception as e:
                err_msg = json.dumps({"type": "status", "content": f"❌ Ingestion error: {str(e)}"})
                yield f"data: {err_msg}\n\n"
                logger.error(f"Web doc ingestion error: {e}")
            
            # Send summary to LLM
            summary_msg = f"I just pasted a document ({line_count} lines). Please acknowledge that you've received it."
            try:
                response_stream = orchestrator.process_message(
                    user_input=summary_msg,
                    session_id=session_id,
                    history=history,
                    system_prompt=system_prompt,
                    skip_extraction=True
                )
                for chunk in response_stream:
                    chunk_data = json.dumps({"type": "content", "content": chunk})
                    yield f"data: {chunk_data}\n\n"
                
                history.append({"role": "user", "content": f"[Document pasted: {line_count} lines]"})
            except Exception as e:
                err_data = json.dumps({"type": "error", "content": str(e)})
                yield f"data: {err_data}\n\n"
                logger.error(f"Web LLM error after doc ingest: {e}")
        
        # ── Normal conversation ───────────────────────────────
        else:
            try:
                full_response = ""
                response_stream = orchestrator.process_message(
                    user_input=user_message,
                    session_id=session_id,
                    history=history,
                    system_prompt=system_prompt
                )
                for chunk in response_stream:
                    chunk_data = json.dumps({"type": "content", "content": chunk})
                    yield f"data: {chunk_data}\n\n"
                    full_response += chunk
                
                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": full_response})
            except Exception as e:
                err_data = json.dumps({"type": "error", "content": str(e)})
                yield f"data: {err_data}\n\n"
                logger.error(f"Web orchestrator error: {e}")
        
        # Signal end of stream
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.route("/api/session/<session_id>/save", methods=["POST"])
def save_session(session_id: str):
    """Save a session to disk."""
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    try:
        path = orchestrator.memory_manager.save_session(session_id, session["history"])
        return jsonify({"status": "saved", "path": str(path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/memories")
def debug_memories():
    """Debug endpoint: list all stored facts in Qdrant knowledge_collection.
    
    Usage: GET http://localhost:5000/api/debug/memories
    """
    try:
        from core.config import QDRANT_KNOWLEDGE_COLLECTION, QDRANT_DOCUMENT_COLLECTION
        
        # Get facts from knowledge_collection
        facts_points, _ = orchestrator.memory_manager.retriever.qdrant_manager.client.scroll(
            collection_name=QDRANT_KNOWLEDGE_COLLECTION,
            limit=100,
            with_payload=True
        )
        facts = []
        for p in facts_points:
            payload = p.payload or {}
            facts.append({
                "id": str(p.id),
                "subject": payload.get("subject", ""),
                "predicate": payload.get("predicate", ""),
                "object": payload.get("object", ""),
                "status": payload.get("status", ""),
                "text": payload.get("text", ""),
                "timestamp": payload.get("timestamp", ""),
            })
        
        # Get chunks from document_collection
        chunk_points, _ = orchestrator.memory_manager.retriever.qdrant_manager.client.scroll(
            collection_name=QDRANT_DOCUMENT_COLLECTION,
            limit=100,
            with_payload=True
        )
        chunks = []
        for p in chunk_points:
            payload = p.payload or {}
            chunks.append({
                "id": str(p.id),
                "heading": payload.get("heading", ""),
                "text_preview": payload.get("text", "")[:200],
                "char_count": payload.get("char_count", 0),
                "source_label": payload.get("source_label", ""),
            })
        
        return jsonify({
            "total_facts": len(facts),
            "total_chunks": len(chunks),
            "knowledge_collection": QDRANT_KNOWLEDGE_COLLECTION,
            "document_collection": QDRANT_DOCUMENT_COLLECTION,
            "facts": facts[:20],
            "chunks": chunks[:20],
        })
    except Exception as e:
        logger.error(f"Debug memories endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/retrieve")
def debug_retrieve():
    """Debug endpoint: run full retrieval pipeline for a query without calling LLM.
    
    Usage: GET http://localhost:5000/api/debug/retrieve?q=what+is+my+name
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing 'q' query parameter"}), 400
    
    try:
        # Step 1: Route
        collection = orchestrator.memory_manager.query_router.route(query)
        
        # Step 2: Retrieve
        retrieval_data = orchestrator.memory_manager.retrieve_context(query, n_results=10)
        
        # Step 3: Build context (without LLM call)
        system_prompt = f"You are {AI_NAME} {AI_HINT}.\nKeep answers concise and efficient.\n"
        full_context, temperature = orchestrator.context_builder.build_context(
            query, [], system_prompt=system_prompt
        )
        
        # Extract system prompt from built context
        sys_prompt_content = ""
        if full_context and full_context[0].get("role") == "system":
            sys_prompt_content = full_context[0]["content"]
        
        return jsonify({
            "query": query,
            "routed_collection": collection,
            "retrieval_type": retrieval_data["type"],
            "documents": retrieval_data["documents"],
            "document_count": len(retrieval_data["documents"]),
            "temperature": temperature,
            "system_prompt_preview": sys_prompt_content[:500],
            "system_prompt_length": len(sys_prompt_content),
        })
    except Exception as e:
        logger.error(f"Debug retrieve endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/runtime-metrics")
def debug_runtime_metrics():
    """Summarize runtime replay metrics from logs/runtime_replay.jsonl."""
    try:
        entries = load_runtime_entries("logs/runtime_replay.jsonl")
        return jsonify(evaluate_runtime_metrics(entries))
    except Exception as e:
        logger.error(f"Debug runtime metrics endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/citation-metrics")
def debug_citation_metrics():
    """Summarize citation usage from logs/citations.jsonl."""
    try:
        entries = load_citation_entries("logs/citations.jsonl")
        total = len(entries)
        with_citations = sum(1 for entry in entries if entry.get("citations"))
        citation_count = sum(len(entry.get("citations", [])) for entry in entries)
        by_type = {}
        for entry in entries:
            for citation in entry.get("citations", []):
                key = citation.get("memory_type", "unknown")
                by_type[key] = by_type.get(key, 0) + 1
        return jsonify({
            "total_responses": total,
            "responses_with_citations": with_citations,
            "citation_coverage": round(with_citations / total, 4) if total else 0.0,
            "average_citations_per_response": round(citation_count / total, 4) if total else 0.0,
            "citations_by_type": by_type,
        })
    except Exception as e:
        logger.error(f"Debug citation metrics endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Startup ──────────────────────────────────────────────────────────────

def init_app():
    """Initialize the ENML Orchestrator (called once)."""
    global orchestrator, doc_ingester
    print("Initializing ENML Orchestrator for Web Server...")
    orchestrator = Orchestrator()
    doc_ingester = DocumentIngester(orchestrator.memory_manager, llm_client=orchestrator.client)
    print(f"✅ ENML Web Server ready — Orchestrator initialized")


if __name__ == "__main__":
    init_app()
    print(f"\n╔════════════════════════════════════════════════════════════╗")
    print(f"║              ENML Web Chat UI                             ║")
    print(f"╚════════════════════════════════════════════════════════════╝")
    print(f"  URL:     http://localhost:{WEB_SERVER_PORT}")
    print(f"  AI:      {AI_NAME}")
    print(f"  Memory:  Full ENML pipeline (extraction + knowledge graph)")
    print()
    
    app.run(
        host="0.0.0.0",
        port=WEB_SERVER_PORT,
        debug=False,
        threaded=True
    )
