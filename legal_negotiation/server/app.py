import sys
import os

# ── Make sure Python can find models.py in the root /app folder ───────────────
sys.path.insert(0, "/app")

from openenv.core.env_server import create_app
from models import LegalNegotiationAction, LegalNegotiationObservation
from server.legal_negotiation_environment import LegalNegotiationEnvironment

app = create_app(
    LegalNegotiationEnvironment,
    LegalNegotiationAction,
    LegalNegotiationObservation,
    env_name="legal-negotiation",
)