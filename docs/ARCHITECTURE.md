# ENML System Architecture & Workflow

This document provides a technical deep-dive into the External Neural Memory Layer (ENML), explaining its components, storage layers, and the exact step-by-step workflow of how it processes user interactions to form a continuous, intelligent memory stream.

---

## 1. System Overview

ENML is designed as a modular pipeline that intercepts communications between the user and the LLM. It extracts, organizes, retrieves, and distills memories before they ever reach the text generation phase.

```mermaid
graph TD
    User((User Input)) -->|Message| Orchestrator[Orchestrator]
    
    subgraph "Phase 1: Extraction & Identity"
        Orchestrator --> Extractor[Memory Extractor]
        Extractor -->|JSON Triples| Linker[Entity Linker]
        Linker --> AuthMem[Authority Memory]
        Linker --> VectorDB[(Qdrant Vector DB)]
    end
    
    subgraph "Phase 2: Intelligent Retrieval"
        Orchestrator --> Router[Query Router]
        Router -->|Intent Topic| Retriever[Hybrid Retriever]
        Retriever -->|Dense + Sparse| VectorDB
        Retriever -->|Raw Matches| Reranker[CrossEncoder Reranker]
    end
    
    subgraph "Phase 3: Distillation & Injection"
        Reranker -->|Top N Memories| Distiller[Context Distiller]
        Distiller -->|Compressed Summary| ContextBuilder[Context Builder]
        AuthMem -->|Permanent Profile| ContextBuilder
    end
    
    subgraph "Phase 4: Generation"
        ContextBuilder -->|Final Prompt| LLM[Llama.cpp Server]
        LLM -->|Stream| User
    end
```

---

## 2. Core Components

| Component | Responsibility |
|---|---|
| **Memory Extractor** | Uses an LLM to read user input and extract structured JSON facts (Subject-Predicate-Object). Performs validation to reject garbled data. |
| **Authority Memory** | Deterministic JSON storage for the AI's identity and the User's core identity to prevent data drift or hallucination. |
| **Hybrid Retriever** | Polls Qdrant using both Dense Vectors (semantic meaning) and BM25 Sparse Vectors (keyword matching). Applies Memory Aging to penalize very old, low-confidence facts. |
| **CrossEncoder Reranker** | Takes the broad results from the Retriever and strictly scores them based on relevance to the exact query, elevating the most critical memories. |
| **Query Router** | Fast intent classification. Determines if the user is asking a personal question, researching a document, or discussing project code, routing the retrieval to the correct silo. |
| **Context Distiller** | Compresses noisy, fragmented retrieved memories into a dense, concise summary block to save context window space and prevent LLM confusion. |
| **Orchestrator** | The central engine that ties everything together and handles Episodic Conversation Summaries (chunking long conversation histories). |

---

## 3. Live Flow Example: "The Pet Protocol"

To truly understand ENML's architecture, let's walk through a live example. Imagine a user interacting with ENML across two different days.

### Day 1: The Initial Fact

**User says:** *"My name is Flex and I got a new pet lizard today, its name is Colu. Also, I'm working on the ENML project in Python."*

#### Step 1: Fact Extraction (Background)
The `Orchestrator` receives the message and sends it to the `MemoryExtractor`. A fast background LLM parses it into semantic triples:
1. `{"subject": "user", "predicate": "has_name", "object": "Flex", "confidence": 0.99, "type": "identity"}`
2. `{"subject": "user", "predicate": "has_pet", "object": "lizard", "confidence": 0.95, "type": "fact"}`
3. `{"subject": "lizard", "predicate": "has_name", "object": "Colu", "confidence": 0.90, "type": "identity"}`
4. `{"subject": "user", "predicate": "has_project", "object": "ENML", "confidence": 0.95, "type": "project"}`
5. `{"subject": "user", "predicate": "has_skill", "object": "Python", "confidence": 0.90, "type": "skill"}`

#### Step 2: Routing & Storage
The `MemoryManager` routes these facts:
* `"has_name Flex"` hits the `AuthorityMemory` limit guard. It is saved directly to `identity.json` for permanent absolute priority.
* The pet and project facts are sent to the `EntityLinker`. Vectors are generated using the local `BGE-base-en` embedding model.
* The `Retriever` inserts these as points into `Qdrant` along with Sparse BM25 keywords and timestamp metadata.

#### Step 3: Standard Response
The system responds normally: *"Hello Flex! Congratulations on your new lizard, Colu. How's the ENML project coming along?"*

---

### Day 2: The Recall

24 hours later, the user opens a new session.

**User says:** *"Can you write a Python script that prints a greeting for my lizard?"*

#### Step 1: Query Intent Routing
The `Orchestrator` asks the `QueryRouter` to classify the intent. The router recognizes this as a combination of personal knowledge and project discussion. It targets the `knowledge_collection` while allowing general fallback.

#### Step 2: Hybrid Retrieval
The `Retriever` queries Qdrant using the text *"Can you write a Python script that prints a greeting for my lizard?"*
* **Dense Vectors** find memories related to "pets", "Python", and "writing code".
* **Sparse BM25** looks for exact keyword matches for "lizard" and "Python".

Qdrant returns 15 potential memory chunks from the user's history, including the Day 1 facts, plus some random older facts about other animals or languages.

#### Step 3: Memory Aging
The `Retriever` looks at the timestamps. Facts from 3 years ago suffer a minor decay penalty. The fact about the lizard was stored yesterday, so it retains its high score.

#### Step 4: CrossEncoder Reranking
The 15 retrieved facts are paired with the user's query and passed through the `BAAI/bge-reranker-base` model. 
* The reranker gives a massive relevance score to the fact `{"subject": "lizard", "predicate": "has_name", "object": "Colu"}`.
* Irrelevant retrieved facts are discarded. The top 5 facts remain.

#### Step 5: Context Distillation
Instead of pasting 5 separate raw JSON facts into the LLM prompt, ENML sends the top 5 facts to the `ContextDistiller`.
The Distiller compresses them into a dense string:
> *User's pet lizard is named Colu. User codes in Python.*

#### Step 6: Prompt Construction
The `ContextBuilder` constructs the final invisible system prompt:
```text
System Time: 2026-03-10T09:34:50

Your identity:
Name: Jarvis

User's identity:
Name: Flex

Relevant Graph Memory & Context:
User's pet lizard is named Colu. User codes in Python.
```

#### Step 7: Final Generation
The `Llama.cpp` server processes the prompt and streams the output directly to the user:
> *"Sure Flex! Here is a simple Python script to greet Colu:"*
> ```python
> print("Hello, Colu the lizard!")
> ```

---

## 4. Hardware Resources & VRAM Offloading

Because ENML relies on multiple localized AI models, it utilizes a Dynamic VRAM reservation system.

1. **Automation Reserve:** `run_server.sh` keeps 2048 MB of VRAM untouched for background automation tools.
2. **Breathing Room:** 500 MB is kept entirely free to prevent OS UI freezing.
3. **Model Budget:** The remaining VRAM is calculated dynamically to determine the `FINAL_NGL` (Number of GPU Layers) offloaded to the main `llama-server`.
4. **CPU Models:** The Embedding (`bge-base-en`) and Reranker models are executed entirely on the CPU via `sentence-transformers` to maximize generation speed for the heavy LLM.
