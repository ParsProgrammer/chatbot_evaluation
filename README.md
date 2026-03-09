# Chatbot Evaluation Pipeline

A modular evaluation system designed to test and benchmark conversational chatbot APIs.  
The tool simulates multi-turn conversations, validates chatbot responses, measures performance, and produces detailed evaluation reports.

This project was built as part of a **Senior Backend Engineer technical challenge** and focuses on designing a **robust, extensible evaluation architecture for LLM-powered systems**.

---

# Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [High-Level Architecture](#high-level-architecture)
- [Architectural Principles](#architectural-principles)
- [Core Components](#core-components)
  - [CLI Orchestrator](#cli-orchestrator)
  - [Dataset Loader](#dataset-loader)
  - [Evaluation Runner](#evaluation-runner)
  - [Chat Client](#chat-client)
  - [Validators](#validators)
  - [Semantic Similarity Engine](#semantic-similarity-engine)
  - [Metrics Aggregator](#metrics-aggregator)
  - [Reporting Layer](#reporting-layer)
- [Execution Flow](#execution-flow)
- [Metrics](#metrics)
- [Dataset Format](#dataset-format)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Assumptions](#assumptions)
- [Trade-offs](#trade-offs)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

# Overview

Modern chatbot systems powered by Large Language Models (LLMs) produce **non-deterministic outputs**. Traditional testing strategies that rely on exact string matching often fail to evaluate these systems reliably.

This project provides a **structured evaluation framework** that:

- Executes **multi-turn conversations**
- Validates chatbot **intent predictions**
- Evaluates **response quality**
- Measures **latency and performance**
- Handles **LLM output variability**
- Generates **structured reports**

The system is designed to be **extensible, maintainable, and suitable for CI/CD integration**.

---

# Problem Statement

The chatbot exposes a REST API endpoint:

```
POST /chat
```

Request:

```json
{
  "user_id": "string",
  "message": "string"
}
```

Response:

```json
{
  "response": "string",
  "intent": "string",
  "confidence": 0.92
}
```

The evaluation tool must:

- simulate conversations
- validate intents
- validate responses
- measure performance
- support multiple test runs
- produce evaluation reports

Because LLM responses vary, the evaluation system must handle **semantic variability rather than exact matching**.

---

# Solution Overview

The system implements an **evaluation pipeline** that processes chatbot conversations and produces structured metrics.

Main responsibilities:

1. Load test datasets
2. Execute multi-turn conversations
3. Validate responses
4. Measure latency
5. Repeat tests for stability
6. Aggregate metrics
7. Generate reports

The architecture separates these responsibilities into modular components to improve maintainability and extensibility.

---

# High-Level Architecture

```
CLI (run_tests.py)
        |
        v
Dataset Loader
        |
        v
Evaluation Runner
        |
        +-------------------+
        |                   |
        v                   v
Chat Client            Validators
        |                   |
        v                   v
 Chatbot API         Semantic Engine
        |
        v
Metrics Aggregator
        |
        v
Reporting Layer
```

---

# Architectural Principles

### Separation of Concerns

Each module has a single responsibility.

| Layer | Responsibility |
|------|------|
CLI | Entry point and configuration |
Dataset Loader | Parsing and validating datasets |
Evaluation Runner | Executing conversations |
Chat Client | API communication |
Validators | Response validation |
Metrics | Aggregation of results |
Reporting | Output generation |

### Modularity

Components are designed to be easily replaceable.

### Extensibility

New validation strategies, metrics, or transport layers can be added without modifying core execution logic.

### LLM-Aware Evaluation

Instead of strict string comparison, the system supports:

- keyword validation
- regex rules
- semantic similarity

This makes evaluation robust for **LLM-generated responses**.

---

# Core Components

## CLI Orchestrator

The CLI entry point (`run_tests.py`) controls the evaluation pipeline.

Responsibilities:

- parse command line arguments
- load datasets
- initialize evaluation components
- run test cases
- generate reports

Example parameters:

```
--dataset
--base-url
--runs
--concurrency
--timeout
--output
```

---

## Dataset Loader

Parses JSON test datasets and converts them into structured models.

Each dataset contains:

- test identifier
- conversation messages
- expected intents
- expected response rules

---

## Evaluation Runner

The evaluation runner executes chatbot conversations.

### Sequential Conversations

Messages within a conversation must be sent sequentially to preserve chatbot context.

```
User -> Turn 1
Bot  -> Response

User -> Turn 2
Bot  -> Response
```

All turns share the same `user_id`.

### Concurrent Test Execution

Different test cases can run concurrently to improve evaluation throughput.

---

## Chat Client

Responsible for communication with the chatbot API.

Responsibilities:

- sending HTTP requests
- measuring response latency
- parsing responses

The abstraction allows different implementations such as:

- HTTP client
- mock client
- fallback client

---

## Validators

Two validation strategies are implemented.

### Intent Validation

Checks whether the predicted intent matches the expected intent.

Strategies include:

- exact matching
- prefix matching
- alias matching
- fuzzy matching

### Response Validation

Evaluates response correctness using:

- keywords
- regex patterns
- semantic similarity

---

## Semantic Similarity Engine

Because LLM responses vary in wording, semantic similarity scoring is used.

Example:

Expected:

```
"You can reset your password in account settings."
```

Response:

```
"Go to your account settings to change your password."
```

Even though wording differs, semantic similarity recognizes equivalent meaning.

---

## Metrics Aggregator

After test execution, results are aggregated into evaluation metrics.

These metrics capture **correctness, response quality, performance, and stability**.

---

# Execution Flow

```
1. CLI loads dataset
2. Runner executes test cases
3. Messages are sent sequentially
4. Chat client calls chatbot API
5. Validators evaluate responses
6. Results collected across runs
7. Metrics aggregated
8. Reports generated
```

---

# Metrics

The evaluation system produces metrics across **accuracy, response quality, performance, and stability**.

## Intent Accuracy

Measures how often the chatbot predicts the correct intent.

Formula:

```
Intent Accuracy = Correct Intent Predictions / Total Turns
```

This metric evaluates the **intent classification capability** of the chatbot.

---

## Response Pass Rate

Measures how often responses satisfy validation rules.

Validation rules may include:

- keyword presence
- regex matching
- semantic similarity

Formula:

```
Response Pass Rate = Valid Responses / Total Responses
```

This metric evaluates **response quality rather than classification correctness**.

---

## Semantic Similarity Score

Because LLM responses vary in wording, semantic similarity is calculated using sentence embeddings.

Score range:

```
0.0 → unrelated
1.0 → identical meaning
```

Example:

Expected:

```
"You can reset your password in account settings."
```

Response:

```
"Go to settings to change your password."
```

The semantic similarity score captures **meaning equivalence even with different wording**.

---

## Latency Metrics

Performance metrics measure chatbot response times.

Collected metrics include:

| Metric | Description |
|------|------|
Average Latency | Mean response time |
p50 Latency | Median response time |
p90 Latency | Slowest 10% of responses |
p99 Latency | Slowest 1% of responses |

These metrics help detect **performance regressions and tail latency issues**.

---

## Stability Metrics

Because LLM systems are **non-deterministic**, tests may run multiple times.

Example:

```
--runs 3
```

The evaluator measures:

### Intent Agreement

How consistently the same intent is predicted across runs.

### Response Agreement

How consistently responses pass validation rules.

---

## Majority Failure Logic

To reduce noise caused by randomness, the system uses **majority voting**.

Example:

| Run | Result |
|----|----|
1 | Pass |
2 | Fail |
3 | Pass |

Final result:

```
Pass (2 / 3 runs succeeded)
```

A test fails only if **more than half the runs fail**.

---

# Dataset Format

Example dataset:

```json
[
  {
    "test_id": "greeting_test",
    "conversation": [
      "Hello",
      "I need help with my account"
    ],
    "expected_intents": [
      "greeting",
      "account_help"
    ],
    "expected_response_keywords": [
      ["hello", "hi", "welcome"],
      ["account", "help", "support"]
    ]
  }
]
```

---

# Installation

Clone the repository:

```
git clone https://github.com/ParsProgrammer/chatbot_evaluation.git
cd chatbot_evaluation
```

Create a virtual environment:

```
python -m venv venv
```

Activate it:

Mac/Linux

```
source venv/bin/activate
```

Windows

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# How to Run

Example command:

```
python run_tests.py \
  --dataset test_cases.json \
  --base-url http://localhost:8080 \
  --runs 3 \
  --output report.json
```

---

# Assumptions

- chatbot supports `/chat` endpoint
- conversation state is maintained using `user_id`
- intent labels are comparable
- semantic similarity approximates response correctness
- test cases are independent

---

# Trade-offs

### Sequential vs Parallel Execution

Conversations must be sequential to preserve context, but different test cases run concurrently for performance.

### Semantic Validation

Semantic similarity improves robustness but introduces additional computational overhead.

### Multi-run Testing

Multiple runs improve reliability but increase evaluation time.

---

# Limitations

Current limitations include:

- no retry logic for network failures
- semantic model loading may increase initial latency
- no historical comparison of reports
- fallback mock server may hide connectivity issues if used incorrectly

---

# Future Improvements

Potential extensions include:

- CI/CD integration
- regression tracking across builds
- evaluation dashboards
- intent confusion matrices
- pluggable validation strategies
- retry and circuit breaker logic
- distributed evaluation execution

---

# Summary

This project demonstrates how to design a **robust evaluation framework for conversational AI systems**.

Key characteristics:

- modular architecture
- multi-turn conversation simulation
- semantic response validation
- stability-aware evaluation
- detailed reporting

The design focuses on **reliability, extensibility, and production readiness**, which are critical when evaluating modern LLM-powered chatbots.
