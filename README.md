# Chatbot Evaluation Pipeline

An automated backend evaluation tool designed to test a conversational chatbot API and detect regressions in performance, quality, stability, and latency.

The system evaluates chatbot behavior across multiple dimensions, going beyond simple accuracy metrics to provide production-grade insights.

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ParsProgrammer/chatbot_evaluation.git
cd chatbot_evaluation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows (PowerShell)**

```bash
venv\Scripts\Activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Tool

Example CLI command:

```bash
python run_tests.py \
  --dataset test_cases.json \
  --base-url http://localhost:8080 \
  --runs 3 \
  --output report.json
```

### CLI Arguments

| Argument     | Description |
|--------------|------------|
| `--dataset`  | Path to JSON test dataset |
| `--base-url` | Base URL of chatbot API |
| `--runs`     | Number of repeated runs per test (handles LLM variability) |
| `--output`   | Output JSON report file |

---

## 🧠 Architecture & Design

### High-Level Architecture

The system is modular and separates concerns clearly:

1. **CLI Layer**  
   Handles argument parsing and execution control.

2. **Dataset Loader**  
   Validates and loads multi-turn conversation test cases.

3. **Chat Client**  
   - Sends sequential messages to `/chat`
   - Maintains conversation state via `user_id`
   - Captures response text, intent, confidence, and latency

4. **Evaluation Engine**  
   - Intent validation (exact match)
   - Keyword-based response validation
   - Semantic similarity validation (fallback)
   - Multi-run aggregation logic
   - Majority-of-runs failure detection

5. **Metrics Aggregator**  
   Computes multi-dimensional metrics:
   - Correctness
   - Semantic quality
   - Calibration
   - Performance
   - Stability

6. **Report Generator**  
   Produces console summary and structured JSON report.

---

## 📊 Evaluation Metrics

Unlike minimal implementations that only report intent accuracy and response pass rate, this tool evaluates across multiple dimensions.

### 1️⃣ Correctness

- **Intent Accuracy**
- **Response Pass Rate** (keyword OR semantic threshold)

Ensures functional correctness of chatbot behavior.

---

### 2️⃣ Semantic Quality

- **Average Semantic Similarity Score**

Evaluates response meaning rather than exact wording, accounting for LLM variability.

---

### 3️⃣ Calibration

- **Average Confidence Score**

Helps detect:
- Overconfident wrong predictions
- Underconfident correct predictions

Important for production reliability and monitoring.

---

### 4️⃣ Performance

- **Average Latency (ms)**
- **Latency Percentiles (p50 / p90 / p99)**

Percentile metrics help detect:
- Tail latency issues
- Sporadic slow responses
- Infrastructure bottlenecks

---

### 5️⃣ Stability (Multi-Run Analysis)

Since LLMs are non-deterministic, the tool measures:

- **Intent Agreement Rate**
- **Response Agreement Rate**

A test fails only if a turn fails in the majority of runs, preventing false regressions caused by natural variability.

---

## 📌 Assumptions Made

- `/chat` endpoint follows the expected contract:
  ```json
  {
    "response": "string",
    "intent": "string",
    "confidence": float
  }
  ```
- Dataset is properly formatted.
- Intent validation uses normalized exact matching.
- LLM outputs may vary between runs.
- Chatbot API is reachable and stable during execution.

---

## ⚖️ Trade-offs Considered

| Decision | Trade-off |
|----------|-----------|
| Sequential execution | Simpler and deterministic but slower than parallel |
| Keyword + semantic validation | Flexible but computationally heavier |
| Majority-of-runs logic | Robust against noise but may mask rare failures |
| Exact intent match | Clear correctness but strict |

The design favors robustness and clarity over premature optimization.

---

## ⚠️ Limitations

- No parallel execution (runtime increases with dataset size)
- No historical regression tracking
- No automatic retry on transient API failures
- Semantic similarity depends on chosen implementation quality
- No persistent metrics storage

---

## 🔮 Future Improvements

- Parallel test execution
- CI/CD integration (GitHub Actions)
- Historical regression tracking
- Visualization dashboard
- Retry mechanism with exponential backoff
- Configurable thresholds via environment variables
- Docker containerization
- Alerting on instability spikes
- Confusion matrix for intent classification

---

## 🧪 Example Files

Included in repository:

- `test_cases.json` → Sample dataset
- `report.json` → Example generated report

---

## 🛠 Dependencies

All dependencies are listed in:

```
requirements.txt
```

---

## 🏗 Engineering Focus

This implementation prioritizes:

- Clear modular structure
- Thoughtful handling of LLM variability
- Meaningful evaluation metrics
- Stability-aware regression detection
- Production-relevant performance insights

The goal is not only to test correctness, but to evaluate behavioral reliability of an LLM-powered chatbot in realistic conditions.

---

## 👤 Author

ParsProgrammer  
GitHub: https://github.com/ParsProgrammer
