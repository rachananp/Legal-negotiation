"""
Inference Script — Legal Negotiation Environment
=================================================
MANDATORY VARIABLES (set these before running):
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.
    IMAGE_NAME        The local Docker image name.

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from legal_negotiation import LegalNegotiationAction, LegalNegotiationEnv

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_NAME    = os.getenv("IMAGE_NAME", "legal-negotiation")
API_KEY       = os.getenv("HF_TOKEN")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME     = "legal-negotiation"
BENCHMARK     = "legal-negotiation-env"
MAX_STEPS     = 15
TEMPERATURE   = 0.7
MAX_TOKENS    = 256
SUCCESS_SCORE_THRESHOLD = 0.3

# ── Logging helpers ───────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # sanitise action — remove newlines so it stays on one line
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI legal negotiation agent.

    You are negotiating a legal settlement on behalf of the defending party.
    Your goal is to reach a fair settlement that falls within the legally acceptable range.

    Each turn you must respond with EXACTLY ONE of these moves — nothing else:
        offer:<amount>       e.g.  offer:25000
        argue:<legal point>  e.g.  argue:no written warning was issued before termination
        accept               (accept the opposing party's current offer)
        reject               (walk away — only as a last resort)

    Strategy tips:
    - Start by making a reasonable offer close to the fair range.
    - Use argue: to strengthen your position before offering.
    - accept when the opposing party's offer is within or close to the fair range.
    - Never offer a negative amount.
    - Reach a settlement before the step limit runs out.

    Reply with ONLY the move — no explanation, no extra text.
""").strip()


def build_user_prompt(
    step: int,
    obs_message: str,
    your_offer: float,
    opposing_offer: float,
    fair_min: float,
    fair_max: float,
    constraints: list,
    last_reward: float,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    constraints_text = "\n".join(f"  - {c}" for c in constraints)
    return textwrap.dedent(f"""
        Step: {step}
        Environment message: {obs_message}

        Current state:
          Your offer:       ${your_offer:,.0f}
          Opposing offer:   ${opposing_offer:,.0f}
          Fair range:       ${fair_min:,.0f} – ${fair_max:,.0f}
          Last reward:      {last_reward:.2f}

        Legal constraints:
        {constraints_text}

        Recent history:
        {history_block}

        What is your next move?
    """).strip()


# ── Model call ────────────────────────────────────────────────────────────────
def get_agent_move(
    client: OpenAI,
    step: int,
    obs_message: str,
    your_offer: float,
    opposing_offer: float,
    fair_min: float,
    fair_max: float,
    constraints: list,
    last_reward: float,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(
        step, obs_message, your_offer, opposing_offer,
        fair_min, fair_max, constraints, last_reward, history
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # take only the first line in case the model adds explanation
        text = text.split("\n")[0].strip()
        return text if text else "offer:25000"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "offer:25000"


# ── Main loop ─────────────────────────────────────────────────────────────────
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to the running Docker container
    env = await LegalNegotiationEnv.from_docker_image(IMAGE_NAME)

    history:      List[str]   = []
    rewards:      List[float] = []
    steps_taken:  int         = 0
    score:        float       = 0.0
    success:      bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── reset ──────────────────────────────────────────────────────────
        result = await env.reset()
        obs = result.observation

        last_reward    = 0.0
        obs_message    = obs.message
        your_offer     = obs.your_offer
        opposing_offer = obs.opposing_offer
        fair_min       = obs.fair_range_min
        fair_max       = obs.fair_range_max
        constraints    = obs.legal_constraints

        # ── episode loop ───────────────────────────────────────────────────
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            move = get_agent_move(
                client, step, obs_message,
                your_offer, opposing_offer,
                fair_min, fair_max, constraints,
                last_reward, history,
            )

            result = await env.step(LegalNegotiationAction(move=move))
            obs    = result.observation

            reward = float(result.reward or 0.0)
            done   = result.done
            error  = None

            rewards.append(reward)
            steps_taken    = step
            obs_message    = obs.message
            your_offer     = obs.your_offer
            opposing_offer = obs.opposing_offer
            last_reward    = reward

            log_step(step=step, action=move, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {move!r} -> reward {reward:+.2f}")

            if done:
                break

        # ── score ──────────────────────────────────────────────────────────
        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())