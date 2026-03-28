"""MLTrainingEnvironment — extends openenv Environment.

Full implementation of reset() and step() with session isolation,
progressive information reveal, and comprehensive error handling.
step() NEVER raises an unhandled exception.
Spec reference: Sections 9, 13, 16.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from typing import Any, Optional

import torch
from openenv.core.env_server.interfaces import Environment

from ml_training_debugger.code_templates import (
    generate_code_snippet,
    validate_fix,
)
from ml_training_debugger.graders import grade_episode
from ml_training_debugger.models import (
    ALL_ACTION_TYPES,
    VALID_CONFIG_KEYS,
    VALID_DIAGNOSES,
    CodeSnippet,
    DataBatchStats,
    EpisodeState,
    MLTrainingAction,
    MLTrainingObservation,
    TrainingConfig,
)
from ml_training_debugger.pytorch_engine import (
    create_model_and_inject_fault,
    extract_gradient_stats,
    extract_model_modes,
    extract_weight_stats,
)
from ml_training_debugger.reward_engine import compute_reward
from ml_training_debugger.scenarios import ScenarioParams, sample_scenario
from ml_training_debugger.simulation import (
    gen_data_batch_stats,
    gen_loss_history,
    gen_val_accuracy_history,
    gen_val_loss_history,
)
from server._baseline_results import store_grader_result

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SessionData:
    """Per-session episode data."""

    scenario: ScenarioParams
    model: torch.nn.Module
    state: EpisodeState
    config: TrainingConfig
    gradient_stats: list[Any]
    weight_stats: list[Any] | None
    model_modes: dict[str, str] | None
    data_batch_stats_raw: dict | None
    code_snippet_raw: dict | None
    loss_history: list[float]
    val_acc_history: list[float]
    val_loss_history: list[float]
    done: bool
    last_score: float | None
    convergence_after_fix: bool


class MLTrainingEnvironment(Environment[MLTrainingAction, MLTrainingObservation, dict]):
    """OpenEnv environment for PyTorch training run debugging.

    Spec Section 9 — Architecture.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sessions: dict[str, SessionData] = {}
        self._last_completed: dict[str, dict] = {}
        self._current_session_id: str = ""

    def _get_session(self, episode_id: str | None = None) -> SessionData | None:
        sid = episode_id or self._current_session_id
        return self._sessions.get(sid)

    def _build_observation(
        self, session: SessionData, reward: float = 0.0
    ) -> MLTrainingObservation:
        """Build observation from session data."""
        state = session.state

        gradient_stats_models = []
        if state.gradients_inspected and session.gradient_stats:
            gradient_stats_models = session.gradient_stats

        weight_stats_models = None
        if state.model_weights_inspected and session.weight_stats is not None:
            weight_stats_models = session.weight_stats

        data_batch = None
        if state.data_inspected and session.data_batch_stats_raw is not None:
            data_batch = DataBatchStats(**session.data_batch_stats_raw)

        model_modes = None
        if state.model_modes_inspected and session.model_modes is not None:
            model_modes = session.model_modes

        code_snippet = None
        if state.code_inspected and session.code_snippet_raw is not None:
            code_snippet = CodeSnippet(**session.code_snippet_raw)

        return MLTrainingObservation(
            run_id=self._current_session_id,
            framework="pytorch",
            epoch=20,
            training_loss_history=session.loss_history,
            val_loss_history=session.val_loss_history,
            val_accuracy_history=session.val_acc_history,
            gradient_stats=gradient_stats_models,
            model_weight_stats=weight_stats_models,
            gpu_memory_used_gb=session.scenario.gpu_memory_used_gb,
            gpu_memory_total_gb=16.0,
            learning_rate=session.config.learning_rate,
            current_config=session.config,
            error_log=session.scenario.error_log,
            data_batch_stats=data_batch,
            model_mode_info=model_modes,
            code_snippet=code_snippet,
            available_actions=state.compute_available_actions(),
            episode_state=state,
            notes=session.scenario.notes,
            done=session.done,
            reward=reward,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MLTrainingObservation:
        """Reset environment for a new episode. Spec Section 13."""
        # Determine task_id — passed via kwargs or defaults to task_001
        task_id = kwargs.get("task_id", "task_001")

        # If called with episode_id that has an active session, terminate it
        session_id = episode_id or str(uuid.uuid4())
        if session_id in self._sessions:
            old = self._sessions[session_id]
            if not old.done:
                score = grade_episode(old.scenario.task_id, old.state, old.scenario)
                self._last_completed[session_id] = {
                    "score": score,
                    "task_id": old.scenario.task_id,
                    "steps": old.state.step_count,
                }
                store_grader_result(
                    session_id, score, old.scenario.task_id, old.state.step_count
                )

        self._current_session_id = session_id

        # Derive deterministic seed
        base_seed = seed if seed is not None else 42
        scenario = sample_scenario(task_id, base_seed)

        # Set torch seed for reproducibility
        torch.manual_seed(scenario.seed)

        # Create real PyTorch model with fault injection
        model, info = create_model_and_inject_fault(scenario)

        # Generate parametric curves
        loss_history = gen_loss_history(scenario)
        val_acc_history = gen_val_accuracy_history(scenario)
        val_loss_history = gen_val_loss_history(scenario)

        # Pre-generate data batch stats
        data_batch_raw = gen_data_batch_stats(scenario)

        # Pre-generate code snippet (for Task 6)
        code_snippet_raw = None
        if scenario.bug_type is not None:
            code_snippet_raw = generate_code_snippet(scenario.bug_type, scenario.seed)

        # Build initial config from scenario
        config = TrainingConfig(
            learning_rate=scenario.learning_rate,
            weight_decay=scenario.weight_decay,
        )

        # Create fresh episode state
        state = EpisodeState()

        session = SessionData(
            scenario=scenario,
            model=model,
            state=state,
            config=config,
            gradient_stats=[],
            weight_stats=None,
            model_modes=None,
            data_batch_stats_raw=data_batch_raw,
            code_snippet_raw=code_snippet_raw,
            loss_history=loss_history,
            val_acc_history=val_acc_history,
            val_loss_history=val_loss_history,
            done=False,
            last_score=None,
            convergence_after_fix=False,
        )

        self._sessions[session_id] = session

        logger.info(
            "reset",
            extra={
                "session_id": session_id,
                "task_id": task_id,
                "scenario_seed": scenario.seed,
            },
        )

        return self._build_observation(session)

    def step(
        self,
        action: MLTrainingAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MLTrainingObservation:
        """Process one agent action. NEVER raises. Spec Sections 13, 16."""
        session = self._get_session()

        # No active episode
        if session is None:
            return MLTrainingObservation(
                done=True,
                reward=0.0,
                error_log="Error: no active episode. Call reset(task_id) first.",
            )

        # Episode already done
        if session.done:
            return self._build_observation(session, reward=0.0)

        state = session.state
        scenario = session.scenario
        action_type = action.action_type

        # Increment step count
        state.step_count += 1

        # Validate action_type is a known type
        if action_type not in ALL_ACTION_TYPES:
            reward = compute_reward(action, state, scenario, is_valid_action=False)
            state.actions_taken.append(f"invalid:{action_type}")
            obs = self._build_observation(session, reward=reward)
            obs.error_log = (
                f"Invalid action_type: {action_type}. "
                f"Valid types: {sorted(ALL_ACTION_TYPES)}"
            )
            return obs

        # Check if action is in available_actions
        available = state.compute_available_actions()
        if action_type not in available:
            reward = compute_reward(action, state, scenario, is_valid_action=False)
            state.actions_taken.append(f"unavailable:{action_type}")
            obs = self._build_observation(session, reward=reward)
            obs.error_log = (
                f"Action '{action_type}' not available. " f"Available: {available}"
            )
            return obs

        # Validate required fields for specific actions
        error = self._validate_action_fields(action)
        if error is not None:
            reward = compute_reward(action, state, scenario, is_valid_action=False)
            state.actions_taken.append(f"malformed:{action_type}")
            obs = self._build_observation(session, reward=reward)
            obs.error_log = error
            return obs

        # Dispatch action
        is_correct_fix: bool | None = None
        convergence = False

        try:
            is_correct_fix, convergence = self._dispatch_action(action, session)
        except Exception as exc:
            logger.error(
                "step_error",
                extra={
                    "session_id": self._current_session_id,
                    "action": action_type,
                    "error": str(exc),
                },
                exc_info=True,
            )
            reward = compute_reward(action, state, scenario, is_valid_action=False)
            obs = self._build_observation(session, reward=reward)
            obs.error_log = f"Internal error processing {action_type}: {exc}"
            return obs

        # Record action
        if action_type == "mark_diagnosed" and action.diagnosis:
            state.actions_taken.append(f"mark_diagnosed:{action.diagnosis}")
        else:
            state.actions_taken.append(action_type)

        # Compute reward
        reward = compute_reward(
            action,
            state,
            scenario,
            is_valid_action=True,
            is_correct_fix=is_correct_fix,
            convergence_confirmed=convergence,
        )

        # Check step limit
        if state.step_count >= scenario.max_steps and not session.done:
            session.done = True

        # Check done
        if session.done:
            score = grade_episode(scenario.task_id, state, scenario)
            session.last_score = score
            self._last_completed[self._current_session_id] = {
                "score": score,
                "task_id": scenario.task_id,
                "steps": state.step_count,
            }
            store_grader_result(
                self._current_session_id, score, scenario.task_id, state.step_count
            )
            logger.info(
                "episode_completed",
                extra={
                    "session_id": self._current_session_id,
                    "task_id": scenario.task_id,
                    "steps": state.step_count,
                    "score": score,
                },
            )

        logger.info(
            "step",
            extra={
                "session_id": self._current_session_id,
                "step_count": state.step_count,
                "action_type": action_type,
                "reward": reward,
            },
        )

        return self._build_observation(session, reward=reward)

    def _validate_action_fields(self, action: MLTrainingAction) -> str | None:
        """Validate required fields for specific actions. Return error or None."""
        if action.action_type == "modify_config":
            if action.target is None or action.value is None:
                return "modify_config requires 'target' and 'value' fields"
            if action.target not in VALID_CONFIG_KEYS:
                return f"Unknown config key: {action.target}. Valid: {sorted(VALID_CONFIG_KEYS)}"

        if action.action_type == "mark_diagnosed":
            if action.diagnosis is None:
                return "mark_diagnosed requires 'diagnosis' field"
            if action.diagnosis not in VALID_DIAGNOSES:
                return (
                    f"Invalid diagnosis: {action.diagnosis}. "
                    f"Valid: {sorted(VALID_DIAGNOSES)}"
                )

        if action.action_type == "fix_code":
            if action.line is None or action.replacement is None:
                return "fix_code requires 'line' and 'replacement' fields"

        return None

    def _dispatch_action(
        self, action: MLTrainingAction, session: SessionData
    ) -> tuple[bool | None, bool]:
        """Dispatch action to handler. Returns (is_correct_fix, convergence)."""
        state = session.state
        scenario = session.scenario
        is_correct_fix: bool | None = None
        convergence = False

        at = action.action_type

        if at == "inspect_gradients":
            if not state.gradients_inspected:
                stats = extract_gradient_stats(session.model, scenario)
                session.gradient_stats = stats
                state.gradients_inspected = True
                # Set gradients_were_normal: True if ALL layers is_exploding=False
                state.gradients_were_normal = all(not s.is_exploding for s in stats)

        elif at == "inspect_data_batch":
            state.data_inspected = True

        elif at == "inspect_model_modes":
            if not state.model_modes_inspected:
                modes = extract_model_modes(session.model)
                session.model_modes = modes
                state.model_modes_inspected = True

        elif at == "inspect_model_weights":
            if not state.model_weights_inspected:
                stats = extract_weight_stats(session.model)
                session.weight_stats = stats
                state.model_weights_inspected = True

        elif at == "inspect_code":
            state.code_inspected = True

        elif at == "modify_config":
            if action.target and action.value is not None:
                setattr(session.config, action.target, action.value)
                state.fix_action_taken = True

        elif at == "add_callback":
            state.fix_action_taken = True

        elif at == "replace_optimizer":
            state.fix_action_taken = True

        elif at == "patch_data_loader":
            state.fix_action_taken = True

        elif at == "fix_model_mode":
            state.fix_action_taken = True

        elif at == "fix_code":
            state.fix_action_taken = True
            if scenario.bug_type and action.line and action.replacement:
                is_correct_fix = validate_fix(
                    scenario.bug_type, action.line, action.replacement
                )
            else:
                is_correct_fix = False

        elif at == "restart_run":
            state.restart_after_fix = True
            # Check convergence — did the fix address the root cause?
            convergence = self._check_convergence(session)
            session.convergence_after_fix = convergence

        elif at == "mark_diagnosed":
            state.diagnosis_submitted = True
            session.done = True

        elif at == "rollback_checkpoint":
            pass  # No-op for now

        return is_correct_fix, convergence

    def _check_convergence(self, session: SessionData) -> bool:
        """Check if the applied fix would resolve the root cause."""
        scenario = session.scenario
        state = session.state
        root = scenario.root_cause.value

        if root == "lr_too_high":
            return (
                "modify_config" in state.actions_taken
                and session.config.learning_rate <= 0.001
            )

        if root == "vanishing_gradients":
            return (
                "modify_config" in state.actions_taken
                and session.config.learning_rate >= 0.001
            )

        if root == "data_leakage":
            return "patch_data_loader" in state.actions_taken

        if root == "overfitting":
            return (
                "modify_config" in state.actions_taken
                or "add_callback" in state.actions_taken
            )

        if root == "batchnorm_eval_mode":
            return "fix_model_mode" in state.actions_taken

        if root == "code_bug":
            return "fix_code" in state.actions_taken and state.fix_action_taken

        return False

    @property
    def state(self) -> dict:
        """Return current environment state."""
        session = self._get_session()
        if session is None:
            return {"status": "no_active_episode"}
        return {
            "status": "active",
            "task_id": session.scenario.task_id,
            "step_count": session.state.step_count,
            "done": session.done,
        }

    def get_last_completed(self, session_id: str | None = None) -> dict | None:
        """Get last completed episode data for grader endpoint."""
        if session_id:
            return self._last_completed.get(session_id)
        # Return most recent
        if self._last_completed:
            return list(self._last_completed.values())[-1]
        return None
