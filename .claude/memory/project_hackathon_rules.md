---
name: Hackathon rules and evaluation criteria
description: Meta PyTorch OpenEnv Hackathon scoring rubric, DQ criteria, and submission requirements.
type: project
---

## Hackathon: Meta PyTorch OpenEnv Hackathon x Scaler School of Technology

**Timeline**: March 14 – April 8, 2026 (Round 1 submission)
**Prize pool**: $30,000
**Top teams advance**: 2,000-3,000 teams to in-person Round 2 (April 25-26, Bangalore)

## Scoring Rubric

| Criterion | Weight |
|-----------|--------|
| Real-world utility | 30% |
| Task & grader quality | 25% |
| Environment design | 20% |
| Code quality & spec compliance | 15% |
| Creativity & novelty | 10% |

## DQ Criteria (auto-fail)
- HF Space doesn't deploy or respond to reset()
- openenv validate fails
- Dockerfile doesn't build
- Baseline doesn't reproduce
- <3 tasks with graders
- Graders always return same score
- No baseline inference script
- Plagiarized environment

## Required Submission Artifacts
1. Public GitHub repo (code, README, requirements, demo script)
2. HF Spaces demo link (tagged `openenv`)
3. README with: env description, action/obs spaces, task descriptions, setup instructions, baseline scores

## Required Endpoints
- `POST /baseline` — trigger inference, return baseline scores
- `POST /grader` — return grader score after completed episode
- `GET /tasks` — return task list with action schema

## Evaluation Phases
1. **Automated Validation**: pass/fail gate (deploy, spec compliance, baseline reproduces)
2. **Agentic Evaluation**: standard Open LLM agent run against all environments
3. **Human Review**: Meta/HF engineers review top submissions

**Why:** Understanding the rubric is essential to prioritize work. Real-world utility (30%) + task quality (25%) = 55% of score. Code quality is only 15%.

**How to apply:** When making trade-offs, prioritize task quality and realism over code perfection. Ensure all DQ criteria pass before polishing.
