from dataclasses import dataclass


@dataclass
class AgentType:
    name: str
    beta: float    # Narrative transmission rate
    gamma: float   # Narrative recovery/exit rate
    weight: float  # Population fraction
    color: str     # Visualization color


# β = narrative adoption speed; γ = narrative abandonment speed.
# Value investors are slow to adopt (low β) but hold their views for months (low γ).
# Contrarians rarely adopt (low β) but exit the moment they do (high γ).
VALUE        = AgentType("Value",         beta=0.08, gamma=0.12, weight=0.30, color="#2196F3")
MOMENTUM     = AgentType("Momentum",      beta=0.65, gamma=0.08, weight=0.25, color="#FF9800")
RETAIL       = AgentType("Retail",        beta=0.80, gamma=0.05, weight=0.20, color="#F44336")
INSTITUTIONAL= AgentType("Institutional", beta=0.35, gamma=0.10, weight=0.15, color="#9C27B0")
CONTRARIAN   = AgentType("Contrarian",    beta=0.03, gamma=0.55, weight=0.10, color="#4CAF50")

DEFAULT_POPULATION: list[AgentType] = [VALUE, MOMENTUM, RETAIL, INSTITUTIONAL, CONTRARIAN]


def population_beta(agents: list[AgentType]) -> float:
    total = sum(a.weight for a in agents)
    return sum(a.beta * a.weight for a in agents) / total


def population_gamma(agents: list[AgentType]) -> float:
    total = sum(a.weight for a in agents)
    return sum(a.gamma * a.weight for a in agents) / total
