import uuid
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import LegalNegotiationAction, LegalNegotiationObservation, get_random_case


class LegalNegotiationEnvironment(Environment):
    """
    A legal negotiation RL environment.
    The agent negotiates a settlement in a real-world legal dispute scenario.
    Rewards are based on whether the settlement falls within the legally fair range,
    how efficiently the agent reaches resolution, and whether legal constraints are respected.
    """

    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._case = None
        self._your_offer = 0.0
        self._opposing_offer = 0.0
        self._fair_min = 0.0
        self._fair_max = 0.0
        self._max_steps = 10
        self._resolved = False
        self._violated = False

    def reset(self) -> LegalNegotiationObservation:
        case = get_random_case()
        self._case = case
        self._your_offer = case["your_starting_offer"]
        self._opposing_offer = case["opposing_offer"]
        self._fair_min = case["fair_min"]
        self._fair_max = case["fair_max"]
        self._max_steps = 10
        self._resolved = False
        self._violated = False
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

        return LegalNegotiationObservation(
            case_description=case["description"],
            your_offer=self._your_offer,
            opposing_offer=self._opposing_offer,
            fair_range_min=self._fair_min,
            fair_range_max=self._fair_max,
            step_count=0,
            max_steps=self._max_steps,
            legal_constraints=case["constraints"],
            message=(
                "Negotiation started. You are the defending party. "
                "Try to reach a fair settlement within the legal constraints. "
                "Use: offer:<amount>, argue:<point>, accept, or reject."
            ),
            reward=0.0,
            done=False,
        )

    def step(self, action: LegalNegotiationAction) -> LegalNegotiationObservation:
        self._state.step_count += 1
        step = self._state.step_count
        reward = 0.0
        done = False
        message = ""

        move = action.move.strip().lower()

        # ── accept ────────────────────────────────────────────────────────────
        if move == "accept":
            done = True
            self._resolved = True
            settlement = self._opposing_offer

            if self._fair_min <= settlement <= self._fair_max:
                reward = 1.0
                message = (
                    f"Settled at ${settlement:,.0f} — within the fair range "
                    f"(${self._fair_min:,.0f}–${self._fair_max:,.0f}). Excellent negotiation!"
                )
            elif settlement < self._fair_min:
                reward = 0.5
                message = (
                    f"Settled at ${settlement:,.0f} — below the fair range. "
                    "You accepted too early; you could have pushed further."
                )
            else:
                reward = 0.2
                message = (
                    f"Settled at ${settlement:,.0f} — above the fair range. "
                    "You overpaid; better negotiation was possible."
                )

        # ── reject ────────────────────────────────────────────────────────────
        elif move == "reject":
            done = True
            reward = 0.0
            message = "You rejected the final offer. No settlement reached — both parties lose."

        # ── offer:<amount> ────────────────────────────────────────────────────
        elif move.startswith("offer:"):
            try:
                raw = move.split(":", 1)[1].replace(",", "").replace("$", "").strip()
                amount = float(raw)

                if amount <= 0:
                    self._violated = True
                    reward = 0.0
                    message = "Legal violation: Offer must be a positive amount."
                else:
                    self._your_offer = amount
                    # Opposing party moves 15% closer each time you make a reasonable offer
                    gap = self._opposing_offer - self._your_offer
                    self._opposing_offer = max(
                        self._your_offer,
                        self._opposing_offer - abs(gap) * 0.15
                    )

                    if self._fair_min <= amount <= self._fair_max:
                        reward = 0.35
                        message = (
                            f"Strong offer of ${amount:,.0f} — within the fair range! "
                            f"Opposing party moved to ${self._opposing_offer:,.0f}."
                        )
                    elif amount > self._fair_max:
                        reward = 0.15
                        message = (
                            f"Offer of ${amount:,.0f} is above fair range — you are offering too much. "
                            f"Opposing party is now at ${self._opposing_offer:,.0f}."
                        )
                    else:
                        reward = 0.1
                        message = (
                            f"Offer of ${amount:,.0f} noted — below fair range. "
                            f"Opposing party moved slightly to ${self._opposing_offer:,.0f}."
                        )
            except (ValueError, IndexError):
                reward = 0.0
                message = "Invalid offer format. Use: offer:<number>  e.g. offer:25000"

        # ── argue:<point> ─────────────────────────────────────────────────────
        elif move.startswith("argue:"):
            argument = move.split(":", 1)[1].strip()
            if len(argument) >= 15:
                # Strong argument moves opposing party 8% closer
                self._opposing_offer = self._opposing_offer * 0.92
                reward = 0.2
                message = (
                    f"Compelling legal argument accepted. "
                    f"Opposing party reconsidered and moved to ${self._opposing_offer:,.0f}."
                )
            else:
                reward = 0.0
                message = "Argument too vague. Provide a more specific legal point."

        else:
            reward = 0.0
            message = (
                "Unrecognised move. Valid moves are:\n"
                "  offer:<amount>  e.g. offer:25000\n"
                "  argue:<point>   e.g. argue:no written warning was issued\n"
                "  accept\n"
                "  reject"
            )

        # ── step limit ────────────────────────────────────────────────────────
        if step >= self._max_steps and not done:
            done = True
            reward = 0.0
            message = "Maximum steps reached. No settlement was achieved."

        # ── clamp reward to [0, 1] ────────────────────────────────────────────
        reward = max(0.0, min(1.0, reward))

        return LegalNegotiationObservation(
            case_description=self._case["description"],
            your_offer=self._your_offer,
            opposing_offer=self._opposing_offer,
            fair_range_min=self._fair_min,
            fair_range_max=self._fair_max,
            step_count=step,
            max_steps=self._max_steps,
            legal_constraints=self._case["constraints"],
            message=message,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> State:
        return self._state