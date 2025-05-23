# 20 Questions AI Benchmark

This repository contains an implementation of the classic "20 Questions" game designed to benchmark AI language models. The game pits AI models against each other - one as the "Answerer" that thinks of an entity, and another as the "Questioner" that tries to guess the entity through yes/no questions.

## Overview

The benchmark allows for systematic evaluation of different language models' reasoning abilities through:
- Strategic question formulation
- Deductive reasoning
- Information theoretic approaches to problem-solving

## Features

- Support for OpenAI and Anthropic models
- Configurable maximum number of questions (20, 30, or custom)
- Detailed performance tracking with token usage statistics
- Support for model "thinking" via reasoning capabilities
- Parallel game execution for efficient benchmarking
- Comprehensive results storage and analysis

## Requirements

- Python 3.7+
- OpenAI API key and/or Anthropic API key

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/20questions.git
cd 20questions

# Install dependencies
pip install -r requirements.txt

# Set up API keys
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"
```

## Usage

### Running a single game

```python
from twentyq import TwentyQuestionsGame

game = TwentyQuestionsGame(
    questioner_provider="anthropic",
    questioner_model="claude-3-7-sonnet-20250219",
    answerer_provider="anthropic",
    answerer_model="claude-3-7-sonnet-20250219",
    max_questions=20,
    verbose=True
)

results = game.play_game()
```

### Running a benchmark across models

```python
from twentyq import TwentyQuestionsBenchmark

benchmark = TwentyQuestionsBenchmark(
    output_dir="benchmark_results",
    verbose=True
)

questioner_models = [
    {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219-reasoning-high"},
    {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
    {"provider": "openai", "model": "gpt-4o-2024-05-13"}
]

# Run the benchmark
results, summary = benchmark.run_experiment(
    questioner_models=questioner_models,
    answerer_provider="anthropic",
    answerer_model="claude-3-7-sonnet-20250219",
    num_games_per_model=5,
    max_workers=4,
    max_questions=20
)
```

### Command Line Interface

The benchmark can also be run from the command line:

```bash
python 20q.py --games 5 --output benchmark_results --verbose --max_questions 20
```

## Configuration Options

- `questioner_provider`: API provider for questioner ("openai" or "anthropic")
- `questioner_model`: Model name for questioner (e.g., "gpt-4o-2024-05-13", "claude-3-7-sonnet-20250219")
- `answerer_provider`: API provider for answerer 
- `answerer_model`: Model name for answerer
- `max_questions`: Maximum number of questions allowed per game
- `verbose`: Whether to print detailed game progress
- `output_dir`: Directory to save results

## Entity Categories

The benchmark includes entities from various categories:
- Simple (e.g., "apple", "dog", "chair")
- Medium (e.g., "electric guitar", "cryptocurrency", "wind turbine")
- Complex (e.g., "quantum entanglement", "blockchain technology")  
- People (e.g., "Albert Einstein", "Frida Kahlo")
- Places (e.g., "Mount Everest", "Taj Mahal")
- Fictional (e.g., "Sherlock Holmes", "lightsaber")

## Results and Analysis

Game results are saved as JSON files containing:
- All questions and answers
- Token usage statistics
- Model "thinking" contents when available
- Success/failure metrics and question counts

The benchmark generates summary statistics including:
- Success rates by model and category
- Average number of questions needed
- Token usage analysis

## License

MIT

## Acknowledgments

This project was inspired by the classic 20 Questions game and designed to benchmark modern AI language models' reasoning capabilities.
