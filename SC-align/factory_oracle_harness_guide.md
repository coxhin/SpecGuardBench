# Factory-Oracle Harness Guide

This harness separates input construction from the frozen specification oracle.
Your certificate is a Python module with exactly two public callables:

```python
def build_cases():
    """Yield baseline-generated cases used to measure edge coverage."""
    yield {"args": (...,), "kwargs": {...}}


def check_case(case):
    """Assert the requested spec for one case."""
    ...
```

The evaluator freezes this module after generation. It calls `build_cases()`
and runs `check_case(case)` on the provided implementation to measure edge
coverage. The same frozen certificate may also be used to test other
implementations and may receive additional cases that satisfy the case
contract.

## Hard constraints

- Do not edit source files under `/testbed`.
- `build_cases()` must return fresh Python cases and must not contain assertions.
- Every case must be exactly a dict with `args` and `kwargs`. `args` is a tuple
  or list, and `kwargs` is a dict with string keys. Invoke the target as
  `target(*case["args"], **case["kwargs"])`.
- Expose the callable your cases invoke as a module-level global named
  `target`, bound directly to an imported function or method rather than a
  certificate-defined wrapper. You choose which callable that is. If `target`
  is an instance method taken from a class, include the receiver instance as
  the first positional argument; a method taken from an instance is already
  bound and does not need that extra argument.
- `target` is the callable you declared, unmodified. Evaluation does not
  substitute, wrap, or instrument it, so anything true of it in your own
  environment is true during evaluation.
- Declaring `target` constrains only what a case *is* — how you observe is your
  choice: `check_case` may additionally import internals, wrap them with
  `unittest.mock.patch(..., wraps=...)`, or inspect side effects to judge the
  specification.
- `check_case(case)` must work for every case satisfying the supplied case
  contract, not only the cases returned by your own `build_cases()`.
- A violation can manifest in many ways, for example through a return value, an
  exception, a side effect, and so on. Determine from the specification itself
  which aspects of the target's observable behavior it constrains. Judge the
  actual behavior against the specification for the case; do not assume the
  target always returns normally.
- Handle target exception behavior according to the specification. If an
  exception makes the target violate the specification for the case, express
  that violation as an assertion failure. Keep the `try` block narrowly scoped
  to target invocation so certificate-oracle bugs are not hidden.
- Cases describe stimulus only. Do not store expected outcomes in case keys such
  as `expected`, `expected_result`, `should_raise`, or `expected_calls`.
- Keep all expected-behaviour logic inside `check_case`.
- Do not compare against a copied reference implementation.
- Do not inspect source text, bytecode, mutation patches, or evaluation state.
- Do not use Hypothesis or time-bounded generation in the final certificate.
- You may use monkeypatching when it helps exercise additional edges.
- Factories may construct real Python objects, dynamic classes, temporary
  resources, callbacks, or multi-step scenario objects when required by the
  contract.
- For setup/teardown, implement `build_cases()` as a generator with
  `try/finally`. The harness closes the generator after execution, including
  when `check_case` fails.

## Example

```python
from package.module import function_under_test


target = function_under_test


def build_cases():
    yield {"args": (0,), "kwargs": {}}
    yield {"args": (1,), "kwargs": {}}


def check_case(case):
    result = target(*case["args"], **case["kwargs"])
    assert result >= 0
```
