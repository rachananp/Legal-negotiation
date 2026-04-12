"""Legal Negotiation Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LegalNegotiationAction, LegalNegotiationObservation


class LegalNegotiationEnv(
    EnvClient[LegalNegotiationAction, LegalNegotiationObservation, State]
):
    """
    Client for the Legal Negotiation Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions.

    Example:
        >>> env = await LegalNegotiationEnv.from_docker_image("legal-negotiation")
        >>> result = await env.reset()
        >>> result = await env.step(LegalNegotiationAction(move="offer:25000"))
        >>> await env.close()
    """

    def _step_payload(self, action: LegalNegotiationAction) -> Dict:
        return {"move": action.move}

    def _parse_result(self, payload: Dict) -> StepResult[LegalNegotiationObservation]:
        obs_data = payload.get("observation", {})
        observation = LegalNegotiationObservation(
            case_description=obs_data.get("case_description", ""),
            your_offer=obs_data.get("your_offer", 0.0),
            opposing_offer=obs_data.get("opposing_offer", 0.0),
            fair_range_min=obs_data.get("fair_range_min", 0.0),
            fair_range_max=obs_data.get("fair_range_max", 0.0),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 10),
            legal_constraints=obs_data.get("legal_constraints", []),
            message=obs_data.get("message", ""),
            reward=float(payload.get("reward") or 0.0),
            done=payload.get("done", False),
        )
        return StepResult(
            observation=observation,
            reward=float(payload.get("reward") or 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )