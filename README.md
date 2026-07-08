# The Throne Room 👑

A multi-agent AI royal council, built with **LangChain** + **LangGraph**, running entirely on **3b local Ollama model**. A user submits a *situation*, five specialist advisor agents each research and deliberate independently over a private RAG knowledge base, and a final **King agent** weighs all five reports to deliver one authoritative decision.

Think: *Game of Thrones* small council, but every advisor actually reads the archives before opening their mouth.

---

## Table of Contents

- [Overview](#overview)
- [The World](#the-world)
- [Architecture](#architecture)
  - [Council-Level Flow](#council-level-flow)
  - [Per-Advisor Pipeline](#per-advisor-pipeline)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [How the RAG Works](#how-the-rag-works)
- [Design Notes / Known Limitations](#design-notes--known-limitations)

---

## Overview

The Throne Room simulates a royal council of **Seven Kingdoms**, ruled from a central Crown City by a King, with each kingdom governed by its own noble house. All world lore — geography, history, houses, religions, economy, military doctrine, law, and intelligence — is hand-written into a set of Markdown files that act as the "Royal Archives."

When a user poses a **situation** (a crisis, dilemma, or event), it is routed through five advisor agents *in series*. Each advisor:

1. Independently researches the situation using only the archives relevant to their office,
2. Produces advice strictly within their domain of expertise,
3. Passes their advice forward.

Once all five advisors have spoken, the **King agent** reviews the situation and all five advices, then renders a single final royal decision — without inventing any facts not present in the situation or the advisor reports.

## The World

- **Setting**: A custom fantasy world ("Aetheris") with 7 kingdoms, a central Crown City, and named ruling houses — inspired by *Game of Thrones*, fully original content.
- **Lore files**: Stored as Markdown under `lore2/`, split into:
  - `common/` — world overview, timeline, geography, kingdoms, capital, noble houses, religions, cultures, languages, calendar, currency, trade — **accessible to every advisor**.
  - Per-advisor specialty folders (see below) — accessible **only** to that advisor.

## Architecture

### Council-Level Flow

Advisors run **in series**, each conditioned only on the original situation (not on each other's advice), and the King synthesizes everything at the end:

```
START
  │
  ▼
Master of Coin  ──▶  Master of Law  ──▶  Lord of Whispers  ──▶  Lord Commander  ──▶  Lord Hand
  │                                                                                     │
  └─────────────────────────────  advice accumulates in shared state  ─────────────────┘
                                                                                          ▼
                                                                                    The King
                                                                                          │
                                                                                          ▼
                                                                                 Final Royal Decision
```

This is implemented as a `StateGraph(KingState)` in `agents/The_king/main.py`, where each node calls into an advisor's independent sub-graph (`report()`), and the King node runs a single `king_chain` (prompt → `ChatOllama`) over all five advices plus the situation.

### Per-Advisor Pipeline

Every advisor (Lord Commander, Lord Hand, Lord of Whispers, Master of Coin, Master of Law) is its **own internal LangGraph**, all built identically:

```
                 ┌──────────────┐
   situation ──▶ │  Breaker Node │  →  3–5 domain-specific subqueries
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
             ┌──▶│ Researcher Node│
             │   └──────┬───────┘
             │          │ tool_calls?
             │     yes  ▼
             │   ┌──────────────┐
             └───│  Tool Node    │  (retrieve_info)
                 └──────────────┘
                        │ no tool_calls (subquery answered)
                        ▼
                 ┌──────────────┐
                 │ Controller    │  advance to next subquery, or...
                 └──────┬───────┘
                        │ all subqueries done
                        ▼
                 ┌──────────────┐
                 │ Advisor Node  │  →  final_answer (advice to the King)
                 └──────────────┘
```

**Step-by-step:**

1. **Breaker node** — an LLM (`ChatOllama`, `llama3.2:3b`) reads the situation and generates 3–5 subqueries scoped strictly to that advisor's specialty (e.g. the Lord Commander only asks military questions, never trade or law). Pronouns are disallowed — every subquery must be self-contained with full place names.
2. **Researcher node ⇄ Tool node loop** — for the current subquery, the researcher LLM (bound to the `retrieve_info` tool) calls the tool exactly once per subquery. The tool performs RAG:
   - First pass: embed all files in the advisor's accessible corpus (common + specialty), rank by cosine similarity against the *situation*, keep the **top 5 files**.
   - Second pass: chunk those 5 files, re-embed, rank by cosine similarity against the *subquery*, keep the **top chunks per file** (score-thresholded and capped).
   - Chunks are merged (de-duplicating overlapping windows) and returned to the LLM, which answers the subquery using only that retrieved content (or replies `not answerable`).
3. **Controller node** — clears the message scratchpad and advances to the next subquery, looping back into the researcher node until all subqueries are answered.
4. **Advisor node** — once all subqueries are resolved, the compiled Q&A pairs become "research notes," which a final domain-specific prompt (e.g. `hand_prompt`, `commander_prompt`) turns into a concise, in-character piece of advice for the King.

Each advisor's graph is compiled and exposed via a `report(situation)` function, which the King's graph calls as a black box.

## Project Structure

```
.
├── agents/
│   ├── The_king/              # King agent: gathers all advice, makes final decision
│   │   ├── main.py            # Top-level StateGraph wiring all 6 nodes together
│   │   ├── func.py            # king_chain (prompt | ChatOllama)
│   │   └── prompts.py         # king_prompt
│   │
│   ├── The_lord_commander/     # Military & national security advisor
│   ├── The_lord_hand/          # Governance & administration advisor
│   ├── The_lord_of_whispers/   # Intelligence & espionage advisor
│   ├── The_master_of_coin/     # Economics & treasury advisor
│   └── The_master_of_law/      # Legal & judicial advisor
│       ├── main.py             # Advisor sub-graph (breaker → researcher ⇄ tools → advisor)
│       ├── prompts.py          # breaker_prompt, research_prompt, <advisor>_prompt
│       ├── RAG_implementation.py  # Embedding, chunking, cosine similarity, retrieval
│       └── tools.py            # retrieve_info tool + LLM chains
│
├── lore2/                      # The Royal Archives (world data, all Markdown)
│   ├── common/                 # Shared world lore, accessible to all advisors
│   ├── lord_commander/         # Military doctrine, logistics, campaigns
│   ├── lord_hand/               # Administration, council affairs, crisis response
│   ├── lord_of_whispers/        # Intelligence analysis, counterintelligence, covert ops
│   ├── master_of_coins/         # Treasury, trade & taxation, public finance
│   └── master_of_law/           # Constitutional law, criminal justice, judicial reform
│
├── sitautions.md               # Example/test situations for the council
|── world image/ 
|     |__world.png              #image to understand the kingdom even better
├── requirements.txt             #it contain all the packages you need to install
└── README.md                       
```

Each advisor folder is structurally identical — only the prompts and the `lore2/<advisor>` data folder differ.

## Tech Stack

| Component            | Choice                                      |
|-----------------------|----------------------------------------------|
| Orchestration          | LangGraph `0.3.24`                          |
| LLM Framework          | LangChain `0.3.19` / LangChain Core `0.3.40` |
| LLM Runtime            | [Ollama](https://ollama.com) (local)         |
| LLM Model              | `llama3.2:3b` (temperature `0.0`) or you can use llama3(bigger model) |
| Embedding Model        | `BAAI/bge-small-en-v1.5` via `sentence-transformers 3.0.1` |
| Similarity Metric      | Cosine similarity (NumPy)                    |
| JSON Repair            | `json_repair 0.61.2` (for parsing breaker output) |
| Language               | Python 3.11                                  |

Other notable installed packages (see full environment for complete list): `langchain-ollama 0.2.3`, `langchain-community 0.3.18`, `langchain-google-genai`, `langchain-groq`, `langchain-mcp-adapters`, `torch 2.12.1`, `transformers 4.46.3`, `numpy 1.26.4`.

## Setup & Installation

1. **Install [Ollama](https://ollama.com/download)** and pull the model used by the project:
   ```bash
   ollama pull llama3.2:3b
   ```
   If your device supports bigger model then please use-
   ```bash
   ollama pull llama3
   ```

2. **Clone the repo and set up a virtual environment (Python 3.11):**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   

4. **Make sure Ollama is running** in the background before starting the app:
   ```bash
   ollama serve
   ```

## Usage

Run the King's council graph:

```bash
python -m agents.The_king.main
```

You'll be prompted:

```
give a situation:
```

Enter a situation (a crisis, event, or dilemma affecting the Seven Kingdoms, if you have no idea you can pick one from situations.md). The council will process it sequentially — Master of Coin → Master of Law → Lord of Whispers → Lord Commander → Lord Hand — with each advisor's advice printed to the console as they "speak" in council, followed by the King's final verdict:

```
King:After hearing the council, the Crown's final decision is:
------> <final royal decision>
```

Example situations to test with are available in `sitautions.md`.

## How the RAG Works

The retrieval system (`RAG_implementation.py`, duplicated per-advisor with different data scopes) uses a two-stage coarse-to-fine retrieval:

1. **File-level retrieval**: Every Markdown file in the advisor's accessible corpus (`common/` + `<advisor>/`) is embedded as a whole document. The situation is embedded and compared via cosine similarity against every file; the **top 5 files** are selected as the relevant working set. Results are cached at the module level per advisor process to avoid recomputing embeddings on every subquery.

2. **Chunk-level retrieval**: The top 5 files are split into overlapping fixed-size word chunks (default: 500 words, 400-word overlap). Each chunk is embedded and compared against the **specific subquery** (not the whole situation). Chunks scoring above a similarity threshold are kept (top ~4 per file), then merged back together — de-duplicating any overlapping words between adjacent chunks — into a small set of coherent passages.

3. These merged passages are handed to the researcher LLM via the `retrieve_info` tool result, and the LLM answers the subquery using *only* that content, or responds `not answerable` if the retrieved passages don't address it.

## Design Notes / Known Limitations

- Advisors run strictly in **series** and do **not** see each other's advice — only the King sees all five reports together. This keeps each advisor's reasoning uncontaminated by the others' opinions.
- The `retrieve_info` tool is expected to be called **exactly once per subquery** — prompts enforce this, but it relies on the small local model (`llama3.2:3b`) following instructions faithfully.
- Message state is explicitly cleared (`RemoveMessage`) between subqueries in the controller node to prevent cross-subquery context contamination.
- Because everything runs on a local 3B parameter model via Ollama, response quality and instruction-following are more variable than with larger hosted models — prompts are written defensively (explicit JSON formatting rules, banned pronouns, strict word limits, etc.) to compensate.
- The King is instructed never to invent facts, ask for clarification, or explain its reasoning — it must issue one direct, in-character decision per situation.
