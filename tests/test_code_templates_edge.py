"""Edge-case tests for code_templates.py — covers AST fallback and tokenizer paths."""

from __future__ import annotations

from ml_training_debugger.code_templates import (
    _normalize_code,
    _tokenize_compare,
    generate_code_snippet,
    validate_fix,
)


class TestNormalizeCode:
    def test_strips_whitespace(self) -> None:
        assert _normalize_code("  model.train()  ") == "model.train()"

    def test_multiline(self) -> None:
        result = _normalize_code("  line1  \n  line2  \n")
        assert "line1" in result
        assert "line2" in result


class TestTokenizeCompare:
    def test_identical_tokens(self) -> None:
        assert _tokenize_compare("model.train()", "model.train()")

    def test_whitespace_ignored(self) -> None:
        assert _tokenize_compare("model.train()", "  model.train()  ")

    def test_different_tokens(self) -> None:
        assert not _tokenize_compare("model.train()", "model.eval()")

    def test_invalid_syntax(self) -> None:
        # Tokenizer returns empty list for invalid syntax
        assert _tokenize_compare("(((", "(((")


class TestValidateFixASTFallback:
    """Tests targeting the AST fallback branch in validate_fix."""

    def test_eval_mode_ast_fallback_with_train_keyword(self) -> None:
        # A replacement that doesn't match exact string or tokenize
        # but passes AST validation (contains 'train', no 'eval')
        result = validate_fix("eval_mode", 5, "model.train()  # fixed mode")
        assert result is True

    def test_detach_loss_ast_without_detach(self) -> None:
        # Replacement without .detach() — should pass AST check
        result = validate_fix(
            "detach_loss", 14, "        loss = criterion(output, batch_y)  # no detach"
        )
        assert result is True

    def test_inplace_relu_ast_without_inplace(self) -> None:
        # Replacement without inplace — should pass AST or semantic check
        result = validate_fix("inplace_relu", 15, "        output = F.relu(output)  # fixed")
        assert result is True

    def test_eval_mode_line_zero_invalid(self) -> None:
        assert not validate_fix("eval_mode", 0, "model.train()")

    def test_detach_loss_syntax_error_rejected(self) -> None:
        # Completely invalid syntax replacement
        assert not validate_fix("detach_loss", 14, "    ((( invalid syntax")

    def test_zero_grad_with_comment(self) -> None:
        # zero_grad with inline comment
        assert validate_fix(
            "zero_grad_missing", 11, "        optimizer.zero_grad()  # clear grads"
        )

    def test_zero_grad_without_keyword(self) -> None:
        # Missing zero_grad keyword entirely
        assert not validate_fix("zero_grad_missing", 11, "        pass")


class TestValidateFixSemanticPatterns:
    """Tests targeting semantic equivalence pattern matching."""

    def test_eval_mode_semantic_train_present(self) -> None:
        # Contains model.train() — semantic pattern match
        assert validate_fix("eval_mode", 5, "model.train()")

    def test_eval_mode_with_eval_keyword_fails(self) -> None:
        # Contains model.eval() — semantic pattern should reject
        assert not validate_fix("eval_mode", 5, "model.eval()")

    def test_detach_loss_criterion_without_detach(self) -> None:
        assert validate_fix(
            "detach_loss", 14, "        loss = criterion(output, batch_y)"
        )

    def test_inplace_relu_without_inplace_flag(self) -> None:
        assert validate_fix("inplace_relu", 15, "        output = F.relu(output)")


class TestGenerateCodeSnippetHints:
    """Test hint generation for code snippets."""

    def test_eval_mode_has_hint(self) -> None:
        snippet = generate_code_snippet("eval_mode")
        assert snippet["hint"] is not None

    def test_detach_loss_has_hint(self) -> None:
        snippet = generate_code_snippet("detach_loss")
        assert snippet["hint"] is not None

    def test_zero_grad_no_hint(self) -> None:
        snippet = generate_code_snippet("zero_grad_missing")
        assert snippet["hint"] is None

    def test_inplace_relu_no_hint(self) -> None:
        snippet = generate_code_snippet("inplace_relu")
        assert snippet["hint"] is None
