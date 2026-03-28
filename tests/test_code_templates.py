"""Test code bug generation and fix validation."""

from __future__ import annotations

import pytest

from ml_training_debugger.code_templates import generate_code_snippet, validate_fix


class TestGenerateCodeSnippet:
    def test_eval_mode(self):
        snippet = generate_code_snippet("eval_mode")
        assert "model.eval()" in snippet["code"]
        assert snippet["filename"] == "train.py"
        assert snippet["line_count"] > 0
        assert len(snippet["imports"]) > 0

    def test_detach_loss(self):
        snippet = generate_code_snippet("detach_loss")
        assert ".detach()" in snippet["code"]

    def test_zero_grad_missing(self):
        snippet = generate_code_snippet("zero_grad_missing")
        assert "zero_grad" not in snippet["code"]

    def test_inplace_relu(self):
        snippet = generate_code_snippet("inplace_relu")
        assert "inplace=True" in snippet["code"]

    def test_unknown_bug_raises(self):
        with pytest.raises(ValueError):
            generate_code_snippet("nonexistent_bug")


class TestValidateFix:
    def test_eval_mode_correct_fix(self):
        assert validate_fix("eval_mode", 5, "model.train()")

    def test_eval_mode_with_whitespace(self):
        assert validate_fix("eval_mode", 5, "  model.train()  ")

    def test_eval_mode_wrong_fix(self):
        assert not validate_fix("eval_mode", 5, "pass")

    def test_detach_loss_correct_fix(self):
        assert validate_fix(
            "detach_loss", 14, "        loss = criterion(output, batch_y)"
        )

    def test_detach_loss_with_trailing_spaces(self):
        assert validate_fix(
            "detach_loss", 14, "        loss = criterion(output, batch_y)   "
        )

    def test_zero_grad_correct_fix(self):
        assert validate_fix("zero_grad_missing", 11, "        optimizer.zero_grad()")

    def test_inplace_relu_correct_fix(self):
        assert validate_fix("inplace_relu", 15, "        output = F.relu(output)")

    def test_wrong_line_number(self):
        assert not validate_fix("eval_mode", 999, "model.train()")

    def test_unknown_bug_type(self):
        assert not validate_fix("nonexistent", 1, "pass")
