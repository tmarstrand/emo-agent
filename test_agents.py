#!/usr/bin/env python3
"""
CLI Test Script for Emotional ADK Agents

Interactive command-line interface to test all three agents:
- Baseline Agent (minimal prompting)
- Negotiation Agent (tactics via prompting)
- Emotional Agent (tactics + VAD tool)

Uses ADK Runner with InMemorySessionService for testing.

Usage:
    python test_agents.py --agent baseline
    python test_agents.py --agent negotiation
    python test_agents.py --agent emotional
    python test_agents.py --compare  # Compare all agents with same message
"""
import asyncio
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import os
import warnings
import logging
from typing import Optional

# Suppress Gemini non-text parts warnings
warnings.filterwarnings("ignore")

# Set Vertex AI credentials
# NOTE: Set these environment variables before running, or create a .env file
# See .env.example for template
os.environ.setdefault('GOOGLE_CLOUD_PROJECT', 'your-project-id')
os.environ.setdefault('GOOGLE_CLOUD_LOCATION', 'us-central1')
os.environ.setdefault('GOOGLE_GENAI_USE_VERTEXAI', 'True')

# Disable tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress Google GenAI warnings
logging.getLogger('google.genai').setLevel(logging.ERROR)
logging.getLogger('google').setLevel(logging.ERROR)


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr


from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent_baseline import create_baseline_agent
from agent_negotiation import create_negotiation_agent
from agent_emotional import create_emotional_agent
from tools import analyze_user_emotion


# Agent configurations
AGENTS = {
    "baseline": {
        "factory": create_baseline_agent,
        "description": "Baseline agent with minimal prompting (control)",
    },
    "negotiation": {
        "factory": create_negotiation_agent,
        "description": "Agent with negotiation tactics prompting only",
    },
    "emotional": {
        "factory": create_emotional_agent,
        "description": "Agent with VAD emotion analysis + negotiation tactics",
    },
}

LOG_DIR = Path("conversation_logs")
LOG_DIR.mkdir(exist_ok=True)


class AgentTester:
    """Test harness for emotional agents using ADK Runner."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent_config = AGENTS[agent_name]
        self.agent = self.agent_config["factory"]()
        self.session_service = InMemorySessionService()
        self.app_name = f"emo_agent_{agent_name}"
        self.user_id = "test_user"
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.runner: Optional[Runner] = None
        self.conversation_log = []

    async def initialize(self):
        """Initialize session and runner."""
        # Create session with initial state
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
            state={"turn_count": 0, "conversation_history": []}
        )

        # Create runner
        self.runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service,
        )

        print(f"✓ Initialized {self.agent_name} agent")
        print(f"  Description: {self.agent_config['description']}\n")

    async def send_message(self, user_message: str) -> str:
        """Send message to agent and get response."""
        if not self.runner:
            raise RuntimeError("Tester not initialized. Call initialize() first.")

        # Analyze user emotion (for logging)
        user_vad = analyze_user_emotion(user_message)

        # Prepare message content
        content = types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )

        # Run agent and collect response (suppress warnings)
        response_text = ""
        with suppress_stderr():
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                        # Could also log other parts like thought_signature, function_call here
                        # For now we just extract text for the user-facing response

        # Analyze response VAD (for emotional agent)
        response_vad = None
        if self.agent_name == "emotional" and response_text:
            response_vad = analyze_user_emotion(response_text)

        # Update session state
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id
        )
        if hasattr(session.state, 'to_dict'):
            state = session.state.to_dict()
        else:
            state = dict(session.state)

        state["turn_count"] = state.get("turn_count", 0) + 1

        # Log the exchange
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "turn": state["turn_count"],
            "user_message": user_message,
            "user_vad": user_vad,
            "agent_response": response_text,
        }
        if response_vad:
            log_entry["response_vad"] = response_vad

        self.conversation_log.append(log_entry)

        return response_text, user_vad, response_vad

    def save_log(self):
        """Save conversation log to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"{self.agent_name}_{timestamp}.json"
        with open(log_file, "w") as f:
            json.dump(self.conversation_log, f, indent=2)
        print(f"\n✓ Conversation log saved to {log_file}")


async def interactive_chat(agent_name: str):
    """Interactive chat with a single agent."""
    tester = AgentTester(agent_name)
    await tester.initialize()

    print(f"\n{'='*70}")
    print(f"INTERACTIVE CHAT: {agent_name.upper()}")
    print(f"{'='*70}")
    print("Type your message and press Enter. Type 'quit', 'exit', or 'q' to end.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            # Get response
            response, user_vad, response_vad = await tester.send_message(user_input)

            # Display VAD scores
            print(f"\n[User VAD: V={user_vad['valence']:.2f}, "
                  f"A={user_vad['arousal']:.2f}, "
                  f"D={user_vad['dominance']:.2f}]")

            # Display response
            print(f"\n{agent_name.title()}: {response}")

            # Display response VAD for emotional agent
            if response_vad:
                print(f"\n[Response VAD: V={response_vad['valence']:.2f}, "
                      f"A={response_vad['arousal']:.2f}, "
                      f"D={response_vad['dominance']:.2f}]")

            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

    # Save log
    tester.save_log()


async def compare_agents(message: str):
    """Send the same message to all agents and compare responses."""
    print(f"\n{'='*70}")
    print(f"COMPARING ALL AGENTS")
    print(f"{'='*70}")
    print(f"\nUser message: {message}\n")

    # Analyze user emotion
    user_vad = analyze_user_emotion(message)
    print(f"User VAD: V={user_vad['valence']:.2f}, "
          f"A={user_vad['arousal']:.2f}, "
          f"D={user_vad['dominance']:.2f}\n")

    # Test each agent
    all_logs = []
    for agent_name in AGENTS.keys():
        print(f"--- {agent_name.upper()} ---")

        tester = AgentTester(agent_name)
        await tester.initialize()

        response, _, response_vad = await tester.send_message(message)
        print(f"{response}")

        if response_vad:
            print(f"[Response VAD: V={response_vad['valence']:.2f}, "
                  f"A={response_vad['arousal']:.2f}, "
                  f"D={response_vad['dominance']:.2f}]")
        print()

        all_logs.extend(tester.conversation_log)

    # Save comparison log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"comparison_{timestamp}.json"
    with open(log_file, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"\n✓ Comparison log saved to {log_file}")


async def run_scenarios():
    """Run predefined test scenarios across all agents."""
    scenarios = [
        "I'm so frustrated with this process! Nothing is working!",
        "I'm really excited about this new project!",
        "This is confusing and I don't know what to do...",
        "Why isn't this working? This is ridiculous!",
        "I feel overwhelmed by all these options.",
    ]

    print(f"\n{'='*70}")
    print(f"RUNNING TEST SCENARIOS")
    print(f"{'='*70}\n")

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"SCENARIO {i}/{len(scenarios)}")
        print(f"{'='*70}")
        await compare_agents(scenario)
        await asyncio.sleep(0.5)  # Brief pause between scenarios


def list_agents():
    """List all available agents."""
    print("\nAvailable agents:\n")
    for name, config in AGENTS.items():
        print(f"  {name}")
        print(f"    {config['description']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Test emotional ADK agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat with baseline agent
  python test_agents.py --agent baseline

  # Interactive chat with emotional agent
  python test_agents.py --agent emotional

  # Compare all agents with a specific message
  python test_agents.py --compare --message "I'm so frustrated!"

  # Run predefined test scenarios
  python test_agents.py --scenarios

  # List available agents
  python test_agents.py --list
        """
    )

    parser.add_argument(
        "--agent",
        choices=list(AGENTS.keys()),
        help="Agent to test in interactive mode"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all agents with same message"
    )
    parser.add_argument(
        "--message",
        default="",
        help="Message to send (required with --compare)"
    )
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="Run predefined test scenarios"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available agents"
    )

    args = parser.parse_args()

    if args.list:
        list_agents()
        return

    if args.scenarios:
        asyncio.run(run_scenarios())
        return

    if args.compare:
        if not args.message:
            print("Error: --message is required with --compare")
            parser.print_help()
            return
        asyncio.run(compare_agents(args.message))
        return

    if args.agent:
        asyncio.run(interactive_chat(args.agent))
        return

    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
