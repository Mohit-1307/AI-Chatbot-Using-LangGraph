## 🤖 LangGraph Agent Lab
From Zero to Production-Ready Agentic Workflows
A hands-on, structured learning repository for building agentic AI systems using LangGraph, LangChain, and Ollama.
This project is designed as a mini-course on agents — progressing from stateless graphs to memory, tools, multi-agent systems, and Retrieval-Augmented Generation (RAG).

---

🚀 Why This Repository?

✔ Structured like a real agent engineering course

✔ Focuses on thinking in graphs, not chains

✔ Covers state, tools, memory, and reasoning

✔ Uses local LLMs (Ollama) — no API dependency

✔ Ideal bridge from ML / LLMs → Agentic AI

---

## 🎯 Who Is This For?

This repository is ideal for:

✔️ Developers learning LangGraph

✔️ ML engineers moving into LLM agents

✔️ Engineers building RAG & AI assistants

✔️ Learners who want mental models, not magic

---

## 📚 Learning Outcomes

* After completing this repository, you will be able to:

* Design agent workflows using StateGraph

* Manage long-lived agent state and memory

* Route tool calls safely and deterministically

* Implement ReAct-style reasoning agents

* Build document drafting assistants

* Create production-style RAG pipelines

* Coordinate multiple agents in a system

---

## 🗺️ Learning Path

🟢 Phase 0 – Foundations

1. bot_agent.py – Minimal stateless agent

2. single_input.ipynb – One-shot execution

🟡 Phase 1 – State & Flow

3. memory_agent.py – Persistent conversation memory

4. sequential_input.ipynb – Multi-turn state handling

5. conditional_statements.ipynb – Branching logic

🔵 Phase 2 – Tools & Reasoning

6. react_agent.py – Tool-based reasoning (ReAct)

7. looping_agent.ipynb – Iterative agent execution

🟣 Phase 3 – Multi-Agent Systems

8. multiple_agent.ipynb – Agent coordination

🔴 Phase 4 – Real-World Applications

9. drafter.py – AI document drafting assistant

10. rag_agent.py – PDF-based RAG with ChromaDB

11. exercise_for_graph_*.ipynb – Practice & mastery

---

## 📁 Project Structure

### 1️⃣ bot_agent.py

Minimal single-turn LangGraph agent demonstrating START → PROCESS → END flow with no memory or tools.

### 2️⃣ single_input.ipynb

Notebook version of a one-shot agent execution, useful for visually understanding graph execution.

### 3️⃣ memory_agent.py

Chat agent with persistent conversation memory, storing and reusing past user and AI messages.

### 4️⃣ sequential_input.ipynb

Handles multiple sequential user inputs using the same agent state across turns.

### 5️⃣ conditional_statements.ipynb

Demonstrates conditional routing and branching logic inside LangGraph workflows.

### 6️⃣ react_agent.py

Tool-using ReAct-style reasoning agent with strict tool invocation rules (math tools).

### 7️⃣ looping_agent.ipynb

Shows iterative / looping agent execution, where the agent reasons until a condition is met.

### 8️⃣ multiple_agent.ipynb

Implements multiple agents interacting or coordinating within a single workflow.

### 9️⃣ drafter.py

Tool-based document drafting assistant that can update and save documents autonomously.

### 🔟 rag_agent.py

End-to-end Retrieval-Augmented Generation (RAG) agent using PDF documents and ChromaDB.

### 1️⃣1️⃣ exercise_for_graph_*.ipynb

Hands-on practice exercises to reinforce LangGraph concepts and agent design patterns.

---

### https://git-scm.com/images/logos/downloads/Git-Icon-1788C.png .gitignore

Ignore rules for Python, Jupyter, virtual environments, and system files.

---

## 🛠️ Tech Stack & Tools

* <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="26"/> Python 3.10+

* <img src="https://avatars.githubusercontent.com/u/126733545?s=200&v=4" height="26"/> LangGraph

* <img src="https://avatars.githubusercontent.com/u/126733545?s=200&v=4" height="26"/> LangChain

* <img src="https://avatars.githubusercontent.com/u/151674099?s=200&v=4" height="26"/> Ollama

* <img src="https://avatars.githubusercontent.com/u/103377991?s=200&v=4" height="26"/> ChromaDB

* <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" height="26"/> Jupyter Notebook

---

## ▶️ How to Run

git clone https://github.com/Mohit-1307/LangGraph-Agent-Lab.git

cd LangGraph-Agent-Lab

pip install langchain langgraph langchain-ollama langchain-chroma chromadb

ollama pull llama3.2

ollama pull gemma:2b

ollama pull mxbai-embed-large

---

## 📘 Documentation

📄 README.md

* Explains the purpose and vision of the repository

* Describes the complete folder and notebook structure

* Guides learners on how to follow the learning path step-by-step

* Provides setup instructions and usage guidelines

* Acts as a quick reference for learners, contributors, and recruiters

---

## 🌟 Support & Contribution

If this repository helps you:

⭐ Star the repository
🔁 Share it with fellow learners

Contributions are welcome!
Feel free to open issues or submit pull requests.

---

## 📌 Author

Mohit Singh Rajput 
#### Machine Learning & Data Science Enthusiast
