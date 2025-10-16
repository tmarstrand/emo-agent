"""
Agent 1: Baseline Agent

A simple ADK agent with minimal prompting. Responds naturally to user messages
without any special negotiation tactics or emotional analysis tools.

This serves as a control baseline for comparison with other agents.
"""
from google.adk.agents import Agent


def create_baseline_agent() -> Agent:
    """
    Create a baseline agent with minimal instructions.

    Returns:
        Agent configured with basic conversational abilities
    """
    agent = Agent(
        name="baseline_agent",
        model="gemini-2.5-flash",
        instruction="You are a helpful assistant. Respond naturally and helpfully to user messages.",
    )
    return agent


if __name__ == "__main__":
    # Simple test
    agent = create_baseline_agent()
    print(f"Created {agent.name}")
    print(f"Instruction: {agent.instruction}")
