
import os
import sys
import select
import argparse
from datetime import datetime
from typing import List, Dict

# Add project root to path if needed, though running from root should work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import Orchestrator
from core.logger import get_logger
from core.config import AI_NAME, AI_HINT, MAX_REALTIME_INPUT_CHARS
from core.memory.document_ingester import DocumentIngester

logger = get_logger("ChatInterface")


# ── Multi-line paste reader ──────────────────────────────────────────────
def read_user_input(prompt: str = "\nYou: ") -> str:
    """Read user input, buffering multi-line pastes into a single message.
    
    When the user pastes text, lines arrive on stdin in rapid succession.
    We detect this by checking if more data is available within 100ms after
    each line, and buffer all pasted lines into one string.
    """
    try:
        first_line = input(prompt)
    except EOFError:
        return "exit"
    
    lines = [first_line]
    
    # Check if more lines are immediately available (paste detection)
    while True:
        # select() checks if stdin has data ready to read within timeout
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)  # 100ms timeout
        if ready:
            try:
                line = sys.stdin.readline()
                if line:
                    lines.append(line.rstrip('\n'))
                else:
                    break  # EOF
            except Exception:
                break
        else:
            break  # No more data within timeout — paste is complete
    
    return '\n'.join(lines)


# ── Input classifier ─────────────────────────────────────────────────────
import re

class InputClassifier:
    """Classifies user input as 'conversation' or 'document'.
    
    Documents are large pastes, markdown content, code blocks, etc.
    that should go through batch ingestion instead of real-time extraction.
    """
    
    _DOC_INDICATORS = [
        re.compile(r'^#{1,6}\s+\w', re.MULTILINE),         # Markdown headings
        re.compile(r'```\w*\s*\n', re.MULTILINE),           # Code fences
        re.compile(r'\|[-:]+\|', re.MULTILINE),             # Table separators
        re.compile(r'[┌┐└┘├┤┬┴┼─│═║╔╗╚╝╠╣╦╩╬]'),          # ASCII box-drawing
        re.compile(r'^\s*[-*]\s+\[[ x]\]', re.MULTILINE),   # Task lists
    ]
    
    def classify(self, text: str) -> str:
        """Returns 'conversation' or 'document'."""
        if not text:
            return "conversation"
        
        text = text.strip()
        
        # Length-based: very long inputs are documents
        if len(text) > MAX_REALTIME_INPUT_CHARS:
            # But short multi-line messages (like 2-3 lines) might be normal
            lines = text.split('\n')
            if len(lines) > 5 or len(text) > 1000:
                return "document"
        
        # Structural indicator counting
        indicator_hits = 0
        for pattern in self._DOC_INDICATORS:
            if pattern.search(text):
                indicator_hits += 1
        
        if indicator_hits >= 2:
            return "document"
        
        # High newline density
        lines = text.split('\n')
        if len(lines) > 8:
            avg_line_len = len(text) / max(len(lines), 1)
            if avg_line_len < 60:
                return "document"
        
        return "conversation"


# ── Diagnostics ──────────────────────────────────────────────────────────
def run_diagnostics():
    from core.memory.extractor import RobustJSONParser
    from core.knowledge_graph import EntityLinker
    from core.vector.embeddings import EmbeddingService
    
    print("--- ENML v2.1 Advanced Diagnostics ---")
    
    # 1. Test JSON Regex Parsing
    try:
        parser = RobustJSONParser()
        test_str = "```json\n[{\"subject\": \"user\", \"predicate\": \"like\", \"object\": \"Ubuntu\", \"confidence\": 0.9}]\n```"
        result = parser.parse(test_str)
        assert len(result) == 1
        print("✅ JSON Regex Parsing: Passed")
    except Exception as e:
        print(f"❌ JSON Regex Parsing: Failed - {e}")
        
    # 2. Test Entity Linker Versioning
    try:
        linker = EntityLinker(embedding_service=EmbeddingService())
        # Store Fact 1
        linker.store_fact({"subject": "test_user", "predicate": "has_name", "object": "Old Name", "confidence": 1.0})
        # Store Fact 2 (Contradiction)
        linker.store_fact({"subject": "test_user", "predicate": "has_name", "object": "New Name", "confidence": 1.0})
        print("✅ Entity Linker (Versioning/Contradiction): Passed")
    except Exception as e:
        print(f"❌ Entity Linker: Failed - {e}")
    
    # 3. Test Input Classifier
    try:
        classifier = InputClassifier()
        assert classifier.classify("my name is Flex") == "conversation"
        assert classifier.classify("what is my name?") == "conversation"
        readme_sample = "# Project\n\n## Features\n- Feature 1\n- Feature 2\n\n## Install\n```bash\npip install x\n```\n\n## Usage\nRun the app."
        assert classifier.classify(readme_sample) == "document"
        assert classifier.classify("x" * 1200) == "document"
        print("✅ Input Classifier: Passed")
    except Exception as e:
        print(f"❌ Input Classifier: Failed - {e}")
    
    # 4. Test Document Content Detection
    try:
        from core.memory.extractor import MemoryExtractor
        ext = MemoryExtractor()
        assert ext._is_document_content("## Arch\n\n| Mod | Use |\n|-----|-----|") == True
        assert ext._is_document_content("my name is Flex") == False
        print("✅ Document Content Detection: Passed")
    except Exception as e:
        print(f"❌ Document Content Detection: Failed - {e}")
        
    print("\nDiagnostics complete.")
    sys.exit(0)


# ── Main chat loop ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Flex AI Chat Interface (ENML Powered)")
    parser.add_argument("--session", type=str, help="Resume specific session ID")
    parser.add_argument("--diagnose", action="store_true", help="Run system component tests")
    args = parser.parse_args()
    
    if args.diagnose:
        run_diagnostics()

    print("Initializing ENML Orchestrator...")
    try:
        orchestrator = Orchestrator()
        classifier = InputClassifier()
        doc_ingester = DocumentIngester(orchestrator.memory_manager, llm_client=orchestrator.client)
        logger.info("Orchestrator initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Orchestrator: {e}")
        sys.exit(1)

    # Session Management
    session_id = args.session if args.session else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Session ID: {session_id}")
    
    history: List[Dict[str, str]] = []
    if args.session:
        loaded = orchestrator.memory_manager.get_session(session_id)
        if loaded:
            history = loaded.get("messages", [])
            print(f"Resumed session {session_id} with {len(history)} messages.")

    SYSTEM_PROMPT = (
        f"You are {AI_NAME} {AI_HINT}.\n"
        "Keep answers concise and efficient.\n"
    )

    print(f"\n--- Chat Started (Session: {session_id}) ---")
    print("Type 'exit' to quit, '/remember <text>' to save a fact.")
    print("Paste multi-line content freely — it will be captured as one message.\n")

    while True:
        try:
            user_input = read_user_input().strip()
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit"]:
                print("Saving session and exiting...")
                break

            if user_input.startswith("/remember "):
                fact = user_input.replace("/remember ", "", 1).strip()
                if fact:
                    try:
                        orchestrator.memory_manager.update_profile(fact)
                        print(f"✅ Memory saved: '{fact}'")
                    except Exception as e:
                        print(f"❌ Failed to save memory: {e}")
                        logger.error(f"Remember command failed: {e}")
                continue

            # ── Classify input ────────────────────────────────────
            input_type = classifier.classify(user_input)
            
            if input_type == "document":
                # Batch document ingestion
                line_count = len(user_input.split('\n'))
                print(f"\n📄 Large input detected ({len(user_input)} chars, {line_count} lines) — ingesting as document...")
                
                try:
                    result = doc_ingester.ingest(user_input, source_label="pasted_document")
                    print(f"✅ Document ingested: {result['sections']} sections, "
                          f"{result.get('summaries_stored', 0)} summaries, "
                          f"{result['facts_extracted']} facts extracted")
                except Exception as e:
                    print(f"❌ Document ingestion failed: {e}")
                    logger.error(f"Document ingestion error: {e}")
                
                # Now send a summary to the LLM for a response
                summary_msg = f"I just pasted a document ({line_count} lines). Please acknowledge that you've received it."
                print("AI: ", end="", flush=True)
                full_response = ""
                try:
                    response_stream = orchestrator.process_message(
                        user_input=summary_msg,
                        session_id=session_id,
                        history=history,
                        system_prompt=SYSTEM_PROMPT,
                        skip_extraction=True  # Already handled by DocumentIngester
                    )
                    for chunk in response_stream:
                        print(chunk, end="", flush=True)
                        full_response += chunk
                    print()
                    
                    history.append({"role": "user", "content": f"[Document pasted: {line_count} lines]"})
                    history.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    print(f"\nError: {e}")
                    logger.error(f"LLM response error after doc ingest: {e}")
                continue
            
            # ── Normal conversational message ─────────────────────
            print("AI: ", end="", flush=True)
            full_response = ""
            
            try:
                response_stream = orchestrator.process_message(
                    user_input=user_input, 
                    session_id=session_id, 
                    history=history,
                    system_prompt=SYSTEM_PROMPT
                )
                
                for chunk in response_stream:
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print() 
                
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                print(f"\nError processing message: {e}")
                logger.error(f"Orchestrator Error: {e}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
    if history:
        try:
            path = orchestrator.memory_manager.save_session(session_id, history)
            print(f"Session saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

if __name__ == "__main__":
    main()
