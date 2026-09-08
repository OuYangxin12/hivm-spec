"""hivm-spec: verifier generator for AI-assisted AscendNPU-IR (HIVM) development.

Python-embedded DSL models the HIVM semantic domain (ops, memory spaces,
pipes/events); the generator produces task-specific spec tools (equivalence
checks, UB occupancy views, timelines, ...) that turn MLIR into fast
pass/fail verdicts and runtime-state views. See docs/ for design and roadmap.
"""

__version__ = "0.0.1"
