## Metrics (DSPy)

DSPy is a machine learning framework, so you should define automatic metrics for:
- **Evaluation**: tracking progress over time on a dev set.
- **Optimization**: letting DSPy compile/optimize programs against a target signal.

### What is a metric?

A metric is just a Python function that takes:
- `example`: one datapoint from your dev/train set
- `pred`: the output of your DSPy program
- `trace` (optional): present during compiling/optimization

…and returns a **score** that quantifies output quality.

For simple tasks, metrics can be `accuracy`, `exact match`, `F1`, etc.
For long-form outputs, metrics are usually *smaller programs* that check multiple properties (often using LM feedback).

### Simple metrics

Basic shape:

```python
def metric(example, pred, trace=None):
    return float(...)  # for evaluation/optimization
```

During compiling, DSPy passes `trace != None`. Many teams return a stricter `bool` there:

```python
def metric(example, pred, trace=None):
    score = ...
    if trace is not None:
        return score >= 1.0  # strict gate for bootstrapping
    return score
```

### Evaluation loop

```python
scores = []
for x in devset:
    pred = program(**x.inputs())
    scores.append(metric(x, pred))
```

Or use DSPy’s evaluator:

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=devset, num_threads=4, display_progress=True, display_table=5)
evaluator(program, metric=metric)
```

### This repo: starting-from-scratch metrics

We’re intentionally leaving metrics **empty for now** and will add them iteratively once we’ve
picked what “good output” means for this product.

When we start, prefer something **objective and cheap** (validity/guardrails) first, then iterate
toward usefulness.

### Next iterations (what we typically measure next)

Once validity is stable, add higher-level quality signals:
- **Schema compliance**: required fields per step type, allowed mini types only
- **Relevance**: each step follows from context/answers (no generic filler)
- **Non-redundancy**: avoid asking the same thing twice / restating prior steps
- **Coverage**: hits key required “step ids” / topics for the current phase
- **Tonality**: matches `flow_guide` stage hints (early short/broad → late detailed/pointed)

### Using AI feedback (when outputs are long-form)

For more subjective checks, define a small assessment signature and call it inside your metric:

```python
import dspy

class Assess(dspy.Signature):
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()
```

Then ask targeted questions (correctness, clarity, relevance, etc.) and combine them into a score.

---

## Metrics (Product analytics)

This repo also includes **form analytics** helpers for tracking user behavior across batches/modules
(completion, dropoff, lead capture, etc.).

- Session log shape: `src/metrics/session_log.py`
- Form-wide metrics: `src/metrics/global_metrics.py`
- Per-batch metrics: `src/metrics/batch_metrics.py`
- Optional metrics: `src/metrics/exploratory.py`
- Combined report: `src/metrics/form_analytics.py`
