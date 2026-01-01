## 🤖 LangGraph & Ollama Agents Collection

A hands-on collection of **LangGraph-based AI agents** built with **LangChain** and **Ollama**. This project explores multiple agent patterns such as **single-turn agents, memory-based chat agents, tool-using agents, document drafting agents, and Retrieval-Augmented Generation (RAG)**.

The repository is designed for **learning, experimentation, and practice** with agentic workflows.

---

## 📁 Project Structure

```
.
├── bot_agent.py                  # Basic single-turn LangGraph agent
├── memory_agent.py               # Chat agent with conversation memory
├── drafter.py                    # Document drafting & saving agent (tool-based)
├── rag_agent.py                  # RAG agent using PDF + ChromaDB
├── react_agent.py                # Tool-using reasoning agent (math tools)
├── looping_agent.ipynb           # Looping / iterative agent execution
├── single_input.ipynb            # Single user input agent example
├── sequential_input.ipynb        # Sequential user input handling
├── conditional_statements.ipynb  # Conditional flow in LangGraph
├── multiple_agent.ipynb          # Multiple agents interaction
├── exercise_for_graph_*.ipynb    # LangGraph practice exercises
├── .gitignore                    # Ignore rules for Python, ML & system files
└── README.md                     # Project documentation
```

---

## 🧠 Key Concepts Covered

* LangGraph **StateGraph** fundamentals
 
* Agent state management with **TypedDict**
  
* Message handling (`HumanMessage`, `AIMessage`, `ToolMessage`)
  
* Tool calling & tool routing
  
* Conditional graph execution
  
* Memory persistence across turns
  
* Document drafting and saving via tools
  
* Retrieval-Augmented Generation (RAG)
  
* Vector databases with **ChromaDB**
  
* Ollama local LLM integration

---

## 🧩 Agent Descriptions

### 1️⃣ bot_agent.py

* Minimal **single-turn agent**
* Demonstrates LangGraph basics: `START → PROCESS → END`
* No memory retention

**Use case:** Understanding the simplest LangGraph workflow

---

### 2️⃣ memory_agent.py

* Chat agent with **conversation memory**
* Stores and reuses previous user + AI messages
* Saves full conversation history to a text file

**Use case:** Stateful chatbots

---

### 3️⃣ drafter.py

* Interactive **document drafting assistant**
* Uses tools:
  * `update(content)` → modify document
  * `save(filename)` → persist document
* Agent decides when to update vs save

**Use case:** AI writing assistants, editors

---

### 4️⃣ rag_agent.py

* **RAG pipeline** using a PDF document
* PDF → Chunking → Embeddings → ChromaDB
* Agent retrieves relevant chunks before answering

**Use case:** Question answering over private documents

---

### 5️⃣ react_agent.py

* Tool-using reasoning agent (ReAct-style)
* Math tools:
  * add
  * subtract
  * multiply
* Strict tool usage rules enforced

**Use case:** Controlled reasoning + tool execution

---

### 6️⃣ Jupyter Notebooks

| Notebook                       | Purpose                          |
| ------------------------------ | -------------------------------- |
| `single_input.ipynb`           | One-shot agent execution         |
| `sequential_input.ipynb`       | Sequential inputs handling       |
| `looping_agent.ipynb`          | Continuous looping agent         |
| `conditional_statements.ipynb` | Conditional graph flows          |
| `multiple_agent.ipynb`         | Multiple agents coordination     |
| `exercise_for_graph_*`         | Practice exercises for LangGraph |

---

## ⚙️ Tech Stack

* **Python 3.10+**
* **LangGraph**
* **LangChain**
* **Ollama** (local LLMs)
* **ChromaDB** (vector store)
* **Jupyter Notebook**

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install langchain langgraph langchain-ollama langchain-chroma chromadb
```

### 2. Start Ollama & pull models

```bash
ollama pull llama3.2
ollama pull gemma:2b
ollama pull mxbai-embed-large
```

### 3. Run Python agents

```bash
python bot_agent.py
python memory_agent.py
python drafter.py
python rag_agent.py
```

### 4. Open notebooks

```bash
jupyter notebook
```

---

## 📌 Learning Goals

* Understand agentic workflows
* Build confidence with LangGraph
* Learn tool-based reasoning
* Implement real-world RAG systems
* Practice clean agent architecture

---

## 📄 License

This project is for **educational and experimental purposes**.

---

## 🧠 README.md

* Project purpose and learning objectives

* Overview of LangGraph, LangChain, and Ollama usage

* Agent architecture and workflow patterns

* Explanation of repository structure and file roles

* Description of each agent type and notebook example

* Setup instructions and dependency management

* Recommended learning path for agents and notebooks

* Best practices for experimenting with agentic workflows

---

## ⭐ Tip

If you are learning LangGraph, go **in this order**:

1. `bot_agent.py`
2. `memory_agent.py`
3. `react_agent.py`
4. `drafter.py`
5. `rag_agent.py`
6. Notebooks & exercises

Happy building 🚀

---

## ▶️ How to Run
git clone https://github.com/Mohit-1307/LangGraph-Agent-Lab.git

---

## ⭐ Support

If this repository helps you:

⭐ Star the repo

🔁 Share with learners
