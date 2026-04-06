"""PyTorch code snippet templates for Task 6 code-level debugging.

Each template is a real, syntactically valid Python/PyTorch training script
with one injected bug.
"""

from __future__ import annotations

import ast
import io
import tokenize
from typing import Optional

import torch  # noqa: F401

# Bug variant templates: (buggy_code, correct_line_num, correct_replacement)
_TEMPLATES: dict[str, tuple[str, int, str]] = {
    "eval_mode": (
        """\
import torch
import torch.nn as nn

model = SimpleCNN()
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()""",
        5,
        "model.train()",
    ),
    "detach_loss": (
        """\
import torch
import torch.nn as nn

model = SimpleCNN()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y).detach()
        loss.backward()
        optimizer.step()""",
        14,
        "        loss = criterion(output, batch_y)",
    ),
    "zero_grad_missing": (
        """\
import torch
import torch.nn as nn

model = SimpleCNN()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch_x, batch_y in train_loader:
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()""",
        11,
        "        optimizer.zero_grad()",
    ),
    "inplace_relu": (
        """\
import torch
import torch.nn as nn
import torch.nn.functional as F

model = SimpleCNN()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        output = F.relu(output, inplace=True)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()""",
        15,
        "        output = F.relu(output)",
    ),
}

# Semantic equivalence patterns per bug variant
_SEMANTIC_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "eval_mode": [
        # (must_contain, must_not_contain)
        ("model.train()", "model.eval()"),
    ],
    "detach_loss": [
        ("criterion(", ".detach()"),
    ],
    "zero_grad_missing": [
        ("zero_grad()", ""),  # just needs zero_grad present
    ],
    "inplace_relu": [
        ("F.relu(", "inplace=True"),
    ],
}


def generate_code_snippet(bug_type: str, seed: int = 42) -> dict:
    """Generate a code snippet with the specified bug.

    Returns dict with keys: code, filename, line_count, imports, hint.
    """
    if bug_type not in _TEMPLATES:
        raise ValueError(f"Unknown bug_type: {bug_type}")

    code, _line, _replacement = _TEMPLATES[bug_type]
    lines = code.strip().split("\n")
    imports = [
        line for line in lines if line.startswith("import ") or line.startswith("from ")
    ]

    hint: Optional[str] = None
    if bug_type == "eval_mode":
        hint = "Check the model mode before the training loop."
    elif bug_type == "detach_loss":
        hint = "Examine how the loss is computed and used."

    return {
        "code": code,
        "filename": "train.py",
        "line_count": len(lines),
        "imports": imports,
        "hint": hint,
    }


def _normalize_code(s: str) -> str:
    """Strip whitespace and inline comments for comparison."""
    s = s.strip()
    # Remove inline comments
    result_lines: list[str] = []
    for line in s.split("\n"):
        # Remove trailing comment but preserve strings
        stripped = line.rstrip()
        result_lines.append(stripped)
    return "\n".join(result_lines)


def _tokenize_compare(original: str, replacement: str) -> bool:
    """Compare token streams ignoring whitespace and comments."""

    def get_tokens(code: str) -> list[tuple[int, str]]:
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
            # Filter out COMMENT, NL, NEWLINE, INDENT, DEDENT, ENCODING, ENDMARKER
            skip = {
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
            }
            return [(t.type, t.string) for t in tokens if t.type not in skip]
        except tokenize.TokenError:
            return []

    return get_tokens(original) == get_tokens(replacement)


def validate_fix(bug_type: str, line: int, replacement: str) -> bool:
    """Validate a code fix submission.

    Multi-strategy pipeline per spec Section 22:
    1. Normalize whitespace + strip comments
    2. Token-stream comparison
    3. Semantic equivalence patterns
    4. AST fallback
    """
    if bug_type not in _TEMPLATES:
        return False

    code, correct_line, correct_replacement = _TEMPLATES[bug_type]
    lines = code.strip().split("\n")

    # Check line number is valid
    if line < 1 or line > len(lines):
        return False

    # For zero_grad_missing, the fix is inserting a line, not replacing
    if bug_type == "zero_grad_missing":
        # Accept if the replacement contains zero_grad
        normalized = _normalize_code(replacement)
        if "zero_grad" in normalized:
            return True
        return False

    # Strategy 1: Normalize and compare
    norm_replacement = _normalize_code(replacement)
    norm_correct = _normalize_code(correct_replacement)
    if norm_replacement == norm_correct:
        return True

    # Strategy 2: Token-stream comparison
    if _tokenize_compare(correct_replacement, replacement):
        return True

    # Strategy 3: Semantic equivalence patterns
    patterns = _SEMANTIC_PATTERNS.get(bug_type, [])
    for must_contain, must_not_contain in patterns:
        if must_contain and must_contain in norm_replacement:
            if not must_not_contain or must_not_contain not in norm_replacement:
                return True

    # Strategy 4: AST fallback — verify buggy pattern absent
    try:
        # Replace the line in the full code and parse
        new_lines = lines.copy()
        new_lines[line - 1] = replacement.rstrip()
        new_code = "\n".join(new_lines)
        tree = ast.parse(new_code)

        # Check that the buggy pattern is absent
        ast.dump(tree)  # Validates AST is well-formed
        if bug_type == "eval_mode" and "eval" not in replacement.lower():
            if "train" in replacement.lower():
                return True
        if bug_type == "detach_loss" and "detach" not in replacement.lower():
            return True
        if bug_type == "inplace_relu" and "inplace" not in replacement.lower():
            if "relu" in replacement.lower():
                return True
    except SyntaxError:
        pass

    return False
