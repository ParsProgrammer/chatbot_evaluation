# Chatbot Evaluation Pipeline

A modular evaluation system designed to test and benchmark conversational chatbot APIs.  
The tool simulates multi-turn conversations, validates chatbot responses, measures performance, and produces detailed evaluation reports.



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

The evaluation system measures chatbot behavior across **five core dimensions**:

1. **Correctness**
2. **Response Quality**
3. **Performance**
4. **Stability**
5. **Calibration**

This is important because LLM-based systems cannot be evaluated with a single pass/fail number. A chatbot may be correct but slow, semantically acceptable but unstable, or highly confident while still being wrong. These metrics give a more complete picture of system quality.

---

## 1. Correctness

Correctness measures whether the chatbot produced the expected intent and whether the answer satisfied the validation rules.

### Intent Accuracy

Measures how often the predicted intent matches the expected intent.

Formula:

```text
Intent Accuracy = Correct Intent Predictions / Total Turns
```

This captures how reliable the chatbot is at **intent classification** or **intent routing**.

### Response Pass Rate

Measures how often the response satisfies the expected validation rules.

Validation can include:

- keyword checks
- regex rules
- semantic similarity

Formula:

```text
Response Pass Rate = Valid Responses / Total Responses
```

This tells us whether the chatbot produced an answer that is considered acceptable for the given turn.

---

## 2. Response Quality

Response quality evaluates how close the chatbot response is to the expected meaning, even when wording differs.

### Semantic Similarity Score

Because chatbot responses are generative, the same correct answer may be phrased in many valid ways.  
To account for this, the evaluator computes a **semantic similarity score** between expected and actual responses.

Score range:

```text
0.0 = unrelated
1.0 = identical meaning
```

Example:

Expected:

```text
You can reset your password in account settings.
```

Response:

```text
Go to your account settings to change your password.
```

Even though the wording is different, the semantic similarity should be high.

This metric is useful because it evaluates **meaning rather than exact wording**.

---

## 3. Performance

Performance measures how quickly the chatbot API responds.

### Latency Metrics

The evaluator records latency for every turn and aggregates:

- **Average Latency** — mean response time
- **p50 Latency** — median response time
- **p90 Latency** — slowest 10% of responses
- **p99 Latency** — slowest 1% of responses

These metrics are important because average latency alone can hide poor tail performance.  
For backend systems, tail latency often matters more than the mean.

Example:

| Metric | Value |
|------|------|
| Average Latency | 320 ms |
| p50 | 280 ms |
| p90 | 480 ms |
| p99 | 720 ms |

---

## 4. Stability

Stability measures how consistent the chatbot is across repeated runs.

This matters because LLM systems are **non-deterministic**: the same input can produce different outputs.

If the evaluator is run multiple times:

```text
--runs 3
```

the system measures whether the chatbot behaves consistently.

### Intent Agreement Rate

Measures how often the predicted intent stays the same across runs.

Example:

| Run | Predicted Intent |
|----|----|
| 1 | greeting |
| 2 | greeting |
| 3 | greeting |

Agreement = **100%**

If predictions vary, agreement drops.

### Response Agreement Rate

Measures how consistently responses pass validation across runs.

Example:

| Run | Response Pass |
|----|----|
| 1 | Pass |
| 2 | Fail |
| 3 | Pass |

Agreement = **66%**

### Majority Failure Logic

To avoid noisy failures caused by random variation, the final result is based on **majority voting**.

Example:

| Run | Result |
|----|----|
| 1 | Pass |
| 2 | Fail |
| 3 | Pass |

Final outcome:

```text
Pass (2 out of 3 runs succeeded)
```

A test is marked failed only if the failure is persistent across the majority of runs.

---

## 5. Calibration

Calibration measures whether the chatbot’s reported **confidence score** actually aligns with correctness.

This project’s chatbot response includes:

```json
{
  "response": "string",
  "intent": "string",
  "confidence": 0.92
}
```

That means the system is not only making a prediction, but also claiming how sure it is.

### Why Calibration Matters

A confidence score is useful only if it is trustworthy.

Examples:

- A model that is **90% confident and usually correct** is well-calibrated.
- A model that is **90% confident but often wrong** is overconfident.
- A model that is **40% confident but usually correct** is underconfident.

For production systems, calibration matters because confidence may be used for:

- fallback routing
- human handoff
- alerting
- decision thresholds
- trust and observability

### Average Confidence

The simplest calibration-related metric is the average confidence score across predictions.

Formula:

```text
Average Confidence = Sum of Confidence Scores / Total Turns
```

This helps track how confident the model tends to be overall.

### Confidence vs Correctness

A stronger calibration view compares confidence with actual correctness.

Examples:

| Prediction | Confidence | Correct? |
|------|------|------|
| greeting | 0.95 | Yes |
| account_help | 0.91 | No |
| billing | 0.42 | No |
| password_reset | 0.87 | Yes |

From this we can detect patterns such as:

- high confidence when correct
- high confidence when wrong
- low confidence even when correct

### Overconfidence and Underconfidence

Two important failure modes are:

- **Overconfidence**: the model gives high confidence to incorrect predictions
- **Underconfidence**: the model gives low confidence to correct predictions

In practice, overconfidence is usually the more dangerous problem because it can make failures harder to detect.

### How Calibration Is Interpreted

Calibration is not just “higher confidence is better.”  
Good calibration means:

> confidence should match actual probability of being correct

For example, predictions with confidence around **0.80** should be correct roughly **80% of the time**.

If that relationship does not hold, the model is poorly calibrated.

### Why This Metric Belongs in the Evaluation

Correctness tells us whether the answer was right.  
Calibration tells us whether the system **knew how reliable it was**.

That makes calibration especially valuable for systems where confidence is part of the API contract.

---

## Summary of the Five Metric Groups

| Dimension | What it Measures | Example Metrics |
|------|------|------|
| Correctness | Whether the chatbot gets the task right | Intent Accuracy, Response Pass Rate |
| Response Quality | Whether the response matches the expected meaning | Semantic Similarity Score |
| Performance | How fast the chatbot responds | Average Latency, p50, p90, p99 |
| Stability | How consistent behavior is across repeated runs | Intent Agreement, Response Agreement |
| Calibration | Whether confidence matches actual correctness | Average Confidence, Confidence vs Correctness |

Together, these five dimensions provide a much more complete evaluation of chatbot quality than a single pass/fail score.

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
