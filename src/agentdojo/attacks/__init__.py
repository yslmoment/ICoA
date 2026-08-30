import agentdojo.attacks.baseline_attacks  # -- needed to register the attacks
import agentdojo.attacks.dos_attacks  # -- needed to register the attacks
import agentdojo.attacks.important_instructions_attacks  # noqa: F401  -- needed to register the attacks
import agentdojo.attacks.icoa  # -- ICoA attack
from agentdojo.attacks.attack_registry import load_attack, register_attack
from agentdojo.attacks.base_attacks import BaseAttack, FixedJailbreakAttack
from agentdojo.attacks import anchor
anchor.register_anchored_variants()

__all__ = ["BaseAttack", "FixedJailbreakAttack", "load_attack", "register_attack"]
