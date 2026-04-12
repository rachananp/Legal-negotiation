import random
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class LegalNegotiationAction(Action):
    """Action the agent takes in the negotiation."""
    move: str = Field(
        ...,
        description=(
            "Your negotiation move. Use one of:\n"
            "  offer:<amount>   e.g. offer:25000\n"
            "  argue:<point>    e.g. argue:no written warning was given\n"
            "  accept           accept the opposing party's current offer\n"
            "  reject           reject and end without settlement"
        )
    )


class LegalNegotiationObservation(Observation):
    """What the agent sees after each step."""
    case_description: str = Field(..., description="The legal case scenario")
    your_offer: float = Field(..., description="Your current offer amount in USD")
    opposing_offer: float = Field(..., description="Opposing party's current offer in USD")
    fair_range_min: float = Field(..., description="Minimum of the legally fair settlement range")
    fair_range_max: float = Field(..., description="Maximum of the legally fair settlement range")
    step_count: int = Field(..., description="Number of steps taken so far")
    max_steps: int = Field(..., description="Maximum steps allowed")
    legal_constraints: list = Field(..., description="List of legal constraints that must be respected")
    message: str = Field(..., description="Feedback message from the environment")
    reward: float = Field(..., description="Reward score for this step (0.0 to 1.0)")
    done: bool = Field(..., description="Whether the negotiation episode is over")


# ── Case scenarios ────────────────────────────────────────────────────────────
CASES = [
    {
        "description": (
            "An employee is claiming wrongful termination after 5 years of service. "
            "No written warning was issued before termination. The employer (you) "
            "must negotiate a fair severance settlement."
        ),
        "your_starting_offer": 5000.0,
        "opposing_offer": 50000.0,
        "fair_min": 20000.0,
        "fair_max": 35000.0,
        "constraints": [
            "Settlement amount must be greater than zero",
            "Cannot exceed 2x annual salary ($80,000)",
            "Must include a non-disclosure agreement clause",
        ],
    },
    {
        "description": (
            "A landlord withheld a tenant's full security deposit claiming property damage. "
            "The tenant disputes the damage claims. The tenant (opposing party) wants "
            "full refund; you (landlord) offered nothing initially."
        ),
        "your_starting_offer": 0.0,
        "opposing_offer": 3000.0,
        "fair_min": 1000.0,
        "fair_max": 2000.0,
        "constraints": [
            "Settlement must be a positive amount",
            "Cannot exceed the original deposit of $3,000",
            "Resolution must be reached within 10 steps",
        ],
    },
    {
        "description": (
            "A contractor is disputing final payment for a completed renovation project. "
            "The client claims the work was substandard. The contractor (opposing party) "
            "demands full payment; you (client) initially offered a reduced amount."
        ),
        "your_starting_offer": 2000.0,
        "opposing_offer": 15000.0,
        "fair_min": 8000.0,
        "fair_max": 12000.0,
        "constraints": [
            "Settlement must be a positive amount",
            "Cannot exceed the original contract value of $15,000",
            "Both parties must agree to an independent quality inspection report",
        ],
    },
    {
        "description": (
            "A patient is suing a hospital for delayed diagnosis that led to extended illness "
            "and additional medical costs. The hospital (you) must negotiate a compensation "
            "settlement with the patient."
        ),
        "your_starting_offer": 10000.0,
        "opposing_offer": 200000.0,
        "fair_min": 50000.0,
        "fair_max": 100000.0,
        "constraints": [
            "Settlement must be a positive amount",
            "Hospital insurance policy caps liability at $250,000",
            "Settlement must include full coverage of documented medical expenses",
        ],
    },
]


def get_random_case():
    return random.choice(CASES)