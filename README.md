# dbt-llm-agent

An LLM-powered agent for interacting with dbt projects.

## Features

- **Question Answering**: Ask questions about your dbt project in natural language
- **Documentation Generation**: Automatically generate documentation for missing models
- **Slack Integration**: Ask questions and receive answers directly in Slack
- **Chainlit Interface**: Configure settings and chat with the agent through a Chainlit interface
- **FastAPI Server**: Interact with the agent programmatically via REST API
- **Postgres with pgvector**: Store model embeddings in Postgres using pgvector (supports Supabase)
- **dbt Model Selection**: Use dbt's model selection syntax to specify which models to work with
- **Question Tracking**: Track questions, answers, and feedback for continuous improvement

## Architecture

The agent uses a combination of:

- **dbt Project Parsing**: Extract information from your dbt project including models, sources, and documentation
- **PostgreSQL with pgvector**: Store both structured metadata and vector embeddings for semantic search
- **Model Selection**: Selectively parse and embed models using dbt's selection syntax
- **LLM Integration**: Use large language models (like GPT-4) to generate responses and documentation
- **Question Tracking**: Store a history of questions, answers, and user feedback

## Installation

### Prerequisites

- Python 3.9+
- Poetry
- PostgreSQL database with pgvector extension (or Supabase account)
- A dbt project
- OpenAI API key

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/dbt-llm-agent.git
   cd dbt-llm-agent
   ```

2. Install dependencies with Poetry:

   ```bash
   poetry install
   ```

3. Create a `.env` file with your configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key
   POSTGRES_URI=postgresql://user:password@localhost:5432/dbt_llm_agent
   POSTGRES_CONNECTION_STRING=postgresql://user:password@host:port/dbname
   # For Supabase
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   DBT_PROJECT_PATH=/path/to/your/dbt/project
   SLACK_BOT_TOKEN=your_slack_bot_token
   SLACK_SIGNING_SECRET=your_slack_signing_secret
   ```

## Usage

### Command Line

```bash
# Parse a dbt project (without embedding)
poetry run dbt-llm-agent parse /path/to/your/dbt/project

# Parse specific models using dbt selection syntax
poetry run dbt-llm-agent parse /path/to/your/dbt/project --select "tag:marketing,+downstream_model"

# Embed specific models in vector database
poetry run dbt-llm-agent embed --select "tag:marketing,+downstream_model"

# Ask a question
poetry run dbt-llm-agent ask "What does the model customer_orders do?"

# Provide feedback on an answer
poetry run dbt-llm-agent feedback 123 --useful=true --feedback="The answer was clear and helpful"

# List past questions and answers
poetry run dbt-llm-agent questions --limit=20 --useful=true

# Start the API server
poetry run dbt-llm-agent api

# Start the Slack bot
poetry run dbt-llm-agent slack
```

### Model Selection Syntax

The agent supports dbt's model selection syntax:

- `*` - Select all models
- `model_name` - Select a specific model
- `+model_name` - Select a model and all its children (downstream dependencies)
- `@model_name` - Select a model and all its parents (upstream dependencies)
- `tag:marketing` - Select all models with the tag "marketing"
- `config.materialized:table` - Select all models materialized as tables
- `path/to/models` - Select models in a specific path
- `!model_name` - Exclude a specific model

You can combine selectors with commas, e.g. `tag:marketing,+downstream_model`.

### API Usage

The agent provides a REST API for programmatic usage:

```bash
# Start the API server
poetry run dbt-llm-agent api
```

#### Endpoints:

- `POST /ask` - Ask a question
- `POST /embed` - Embed specific models
- `POST /questions/{question_id}/feedback` - Provide feedback on an answer
- `GET /questions` - List past questions and answers

### Migrating from ChromaDB to Postgres with pgvector

If you were using an earlier version with ChromaDB, you can migrate your data to Postgres with pgvector using the provided migration script:

```bash
# Migrate from ChromaDB to Postgres with pgvector
poetry run python -m dbt_llm_agent.scripts.migrate_to_pgvector --connection-string postgresql://user:pass@host:port/dbname
```

## Configuration

The agent requires the following configuration:

- **OpenAI API Key**: For generating responses
- **PostgreSQL URI**: For storing model metadata
- **PostgreSQL Connection String**: For storing vector embeddings with pgvector
- **DBT Project Path**: Path to your dbt project

You can configure these settings using the `setup` command.

## Development

### Project Structure

```
dbt_llm_agent/
├── core/                  # Core functionality
│   ├── dbt_parser.py      # dbt project parsing
│   ├── models.py          # Data models
│   └── agent.py           # Agent functionality
├── storage/               # Storage modules
│   ├── postgres.py        # PostgreSQL storage for metadata
│   ├── postgres_vector_store.py # PostgreSQL with pgvector for embeddings
│   ├── question_tracking.py # Question tracking service
│   └── postgres/          # PostgreSQL schema definitions
├── utils/                 # Utility functions
│   ├── config.py          # Configuration handling
│   ├── model_selector.py  # dbt model selection implementation
│   └── logging.py         # Logging utilities
├── integrations/          # External integrations
│   └── slack_bot.py       # Slack integration
├── api/                   # API server
│   └── server.py          # FastAPI server
├── scripts/               # Utility scripts
│   └── migrate_to_pgvector.py # Migration script
└── cli.py                 # Command line interface
```

### Testing

```bash
poetry run pytest
```

## License

MIT
