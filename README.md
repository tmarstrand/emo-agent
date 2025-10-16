# EMO AGENT

In standard LLMs there is no direct modelling of emotion. The emotional `feel` you get from the response is due to standard co-occurance of words in the training data.

Additional bias in the answer to `feel` in line with your expectations is via post-training and additional filtering mechanisms. The large LLMs like chatGPT and claude has been post-trained to give the type of answers (tone of voice) that most people find pleasing. And they have all sorts of filters to remove harmful speech, guard against inappropiate answers etc.

This small example is different. It is using direct mapping of the users input into an emotional vector space along the V,A,D axises, Valence, Arousal and Dominance. Once identified the agent then selects a given technique for its answer selected from standard negotiation playbooks:

- Mirroring
- Soft de-escalation
- Reframing
- Orthogonal projection
- Oppositional dampening
- Containment

The idea is to examine if such direct mapping of the users emotional state and explicit exploitation in the response from the LLM can improve the `feel` of the response.

## Why?

I am fascinated by what agents can do in pure process optimization and reasoning. However, to start acting like other than transactional partners agents need to somehow model human emotion.

A lot of human emotion and empathy can be learned and follow standard patterns and rules. The question is to what extent that emotional pattern can be proxied by text interactions, and correctly sensed and acted upon by agents.

This is not a production ready system by any means. It is a test, a sandbox, a small playful thing.

## Structure

The emotional detection is based on the V, A, D (Valence, Arousal, Dominance) system and I use the [EmoBank dataset](https://github.com/JULIELab/EmoBank) for training a simple regression model on sentence embeddings.

This regression model is then applied to any user input, and fed to the agent (programmed in the ADK framework). The agent now have an idea about the VAD projection of the user and a series of guidance rules on how to structure its answer.

The VAD projection is then calculated for the answer of the LLM as well (for documentation).

In multi-turn conversations I store the VAD projections of both the user and the agent for later visualization and inspection.

## Setup

### Prerequisites

- Python 3.10+
- Google Cloud Project (if using Vertex AI)
- uv or pip for package management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emo-agent.git
cd emo-agent
```

2. Install dependencies:
```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

3. Set up Google Cloud credentials (if using Vertex AI):
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your Google Cloud project details
# Or set environment variables directly:
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_GENAI_USE_VERTEXAI=True
```

4. Download the EmoBank dataset:
```bash
# Place emobank.csv in the root directory
# Download from: https://github.com/JULIELab/EmoBank
```

### Training the VAD Model

Train the emotion regression model:
```bash
python train.py
```

This will create `artifacts_v1/` directory with:
- `vad_regressor.pkl` - Ridge regression model
- `vad_scaler.pkl` - StandardScaler for normalization

## Usage

### Testing Agents

Three agent variants are available for comparison:

1. **Baseline Agent**: Minimal prompting (control)
2. **Negotiation Agent**: Tactics via prompting only
3. **Emotional Agent**: VAD tool + negotiation tactics

#### Interactive Chat

Chat with a single agent:
```bash
python test_agents.py --agent emotional
python test_agents.py --agent baseline
python test_agents.py --agent negotiation
```

#### Compare All Agents

Send the same message to all agents:
```bash
python test_agents.py --compare --message "I'm so frustrated!"
```

#### Run Test Scenarios

Run predefined emotional scenarios:
```bash
python test_agents.py --scenarios
```

#### List Agents

```bash
python test_agents.py --list
```

### Output

Conversation logs are saved to `conversation_logs/` with VAD scores:
```json
{
  "timestamp": "2025-10-15T10:30:00",
  "agent": "emotional",
  "turn": 1,
  "user_message": "I'm frustrated!",
  "user_vad": {"valence": 2.1, "arousal": 3.8, "dominance": 2.3},
  "agent_response": "I can see you're frustrated...",
  "response_vad": {"valence": 2.8, "arousal": 2.5, "dominance": 3.1}
}
```

## Project Structure

```
emo-agent/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── emobank.csv              # EmoBank dataset (download separately)
├── train.py                 # Train VAD regression model
├── main.py                  # VADPredictor class
├── tools.py                 # ADK tool wrappers
├── agent_baseline.py        # Baseline agent
├── agent_negotiation.py     # Negotiation tactics agent
├── agent_emotional.py       # Full emotional agent
├── test_agents.py           # Testing harness
└── artifacts_v1/            # Trained models (generated)
    ├── vad_regressor.pkl
    └── vad_scaler.pkl
```

## Dataset Attribution

This project uses the [EmoBank dataset](https://github.com/JULIELab/EmoBank):

> Buechel, S., & Hahn, U. (2017). EmoBank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2017), Valencia, Spain.

## License

MIT License - see LICENSE file for details