"""
CLI interface for dbt-llm-agent
"""

import click
import os
import subprocess
import sys
import logging
from typing import Optional, Any
import pathlib

# Set up logging
from dbt_llm_agent.utils.logging import setup_logging, get_logger, COLORS

# Initialize logging with default settings
setup_logging()
logger = get_logger(__name__)


def colored_echo(text, color=None, bold=False):
    """Echo text with color and styling.

    Args:
        text: The text to echo
        color: Color name from COLORS dict or color code
        bold: Whether to make the text bold
    """
    # Define a mapping of capitalized color names to the COLORS keys
    color_mapping = {
        "BLUE": "INFO",  # Use INFO (green) for BLUE
        "GREEN": "INFO",  # Use INFO (green) for GREEN
        "CYAN": "DEBUG",  # Use DEBUG (cyan) for CYAN
        "RED": "ERROR",  # Use ERROR (red) for RED
        "YELLOW": "WARNING",  # Use WARNING (yellow) for YELLOW
    }

    prefix = ""
    suffix = COLORS["RESET"]

    # Map color name to a color in the COLORS dict if needed
    if color and color in color_mapping and color_mapping[color] in COLORS:
        prefix += COLORS[color_mapping[color]]
    elif color and color in COLORS:
        prefix += COLORS[color]
    elif color:
        prefix += color

    if bold:
        prefix += "\033[1m"

    click.echo(f"{prefix}{text}{suffix}")


def get_env_var(var_name: str, default: Any = None) -> Any:
    """
    Get an environment variable with a default value.

    Args:
        var_name: The name of the environment variable
        default: The default value if the environment variable is not set

    Returns:
        The value of the environment variable or the default value
    """
    return os.environ.get(var_name, default)


def set_logging_level(verbose: bool):
    """Set the logging level based on the verbose flag.

    Args:
        verbose: Whether to enable verbose logging
    """
    # Just set the level of our logger, don't reconfigure logging
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value from environment variables.

    Args:
        key: The configuration key
        default: The default value if the key is not found

    Returns:
        The configuration value or the default value
    """
    env_var = f"{key.upper()}"
    return get_env_var(env_var, default)


@click.group()
def cli():
    """dbt LLM Agent CLI"""
    pass


@cli.command()
def version():
    """Get the version of dbt-llm-agent"""
    colored_echo("dbt-llm-agent version 0.1.0", color="INFO", bold=True)


@cli.command()
@click.argument(
    "project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option(
    "--select",
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
    default=None,
)
@click.option("--force", is_flag=True, help="Force re-parsing of all models")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def parse(project_path, postgres_uri, select, force, verbose):
    """
    Parse a dbt project and store models in the database.

    PROJECT_PATH is the path to the root of the dbt project.
    """
    try:
        # Set logging level based on verbosity
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Load environment variables from .env file (if not already loaded)
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            logger.info("Loaded environment variables from .env file")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Normalize and validate project path
        project_path = pathlib.Path(project_path).resolve()
        if not project_path.exists():
            logger.error(f"Project path does not exist: {project_path}")
            sys.exit(1)

        if not (project_path / "dbt_project.yml").exists():
            logger.error(
                f"Not a valid dbt project (no dbt_project.yml found): {project_path}"
            )
            sys.exit(1)

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage
        from dbt_llm_agent.core.dbt_parser import DBTProjectParser
        from dbt_llm_agent.utils.model_selector import ModelSelector

        # Get PostgreSQL URI from args or env var
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Initialize storage
        logger.info(f"Connecting to PostgreSQL database: {postgres_uri}")
        postgres = PostgresStorage(postgres_uri)

        # Initialize parser
        logger.info(f"Parsing dbt project at: {project_path}")
        parser = DBTProjectParser(project_path)

        # Parse project
        project = parser.parse_project()

        # Create model selector if selection is provided
        if select:
            logger.info(f"Filtering models with selector: {select}")
            selector = ModelSelector(project.models)
            selected_models = selector.select(select)
            logger.info(f"Selected {len(selected_models)} models")

            # Filter project.models to only include selected models
            project.models = {
                name: model
                for name, model in project.models.items()
                if name in selected_models
            }

        # Store models in database
        logger.info(f"Found {len(project.models)} models")
        if force:
            logger.info("Force flag enabled - re-parsing all models")

        for model_name, model in project.models.items():
            if verbose:
                logger.debug(f"Processing model: {model_name}")
            postgres.store_model(model, force=force)

        logger.info(f"Successfully parsed and stored {len(project.models)} models")

        # Store sources if available
        if hasattr(project, "sources") and project.sources:
            logger.info(f"Found {len(project.sources)} sources")
            for source_name, source in project.sources.items():
                if verbose:
                    logger.debug(f"Processing source: {source_name}")
                postgres.store_source(source, force=force)

            logger.info(
                f"Successfully parsed and stored {len(project.sources)} sources"
            )

        return 0

    except Exception as e:
        logger.error(f"Error parsing dbt project: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    "--select",
    required=True,
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option(
    "--postgres-connection-string",
    help="PostgreSQL connection string for vector database",
    envvar="POSTGRES_CONNECTION_STRING",
)
@click.option(
    "--embedding-model", help="Embedding model to use", default="text-embedding-ada-002"
)
@click.option("--force", is_flag=True, help="Force re-embedding of models")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--documentation-only",
    is_flag=True,
    help="Only embed documentation (not interpretation)",
)
@click.option(
    "--interpretation-only",
    is_flag=True,
    help="Only embed interpretation (not documentation)",
)
def embed(
    select,
    postgres_uri,
    postgres_connection_string,
    embedding_model,
    force,
    verbose,
    documentation_only,
    interpretation_only,
):
    """
    Embed selected models in the vector database.

    By default, both documentation and interpretation embeddings are stored if available.
    Use --documentation-only or --interpretation-only to store only one type of embedding.
    """
    try:
        # Set logging level based on verbosity
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            logger.info("Loaded environment variables from .env file")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage
        from dbt_llm_agent.storage.vector_store import PostgresVectorStore
        from dbt_llm_agent.utils.model_selector import ModelSelector

        # Get PostgreSQL URI
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Get PostgreSQL connection string for vector database
        if not postgres_connection_string:
            postgres_connection_string = get_env_var("POSTGRES_CONNECTION_STRING")
            if not postgres_connection_string:
                postgres_connection_string = postgres_uri
                logger.info("Using POSTGRES_URI for vector database connection string")

        # Initialize storage
        logger.info(f"Connecting to PostgreSQL database: {postgres_uri}")
        postgres = PostgresStorage(postgres_uri)

        # Initialize vector store
        logger.info(f"Connecting to vector database: {postgres_connection_string}")
        vector_store = PostgresVectorStore(
            connection_string=postgres_connection_string,
            embedding_model=embedding_model,
        )

        # Get all models from the database
        all_models = postgres.get_all_models()
        logger.info(f"Found {len(all_models)} models in the database")

        # Create model selector
        logger.info(f"Selecting models with selector: {select}")
        models_dict = {model.name: model for model in all_models}
        selector = ModelSelector(models_dict)
        selected_model_names = selector.select(select)

        if not selected_model_names:
            logger.warning(f"No models matched the selector: {select}")
            return 0

        logger.info(f"Selected {len(selected_model_names)} models for embedding")

        # Filter to only selected models
        selected_models = [
            model for model in all_models if model.name in selected_model_names
        ]

        # Validate embedding flags
        if documentation_only and interpretation_only:
            logger.error(
                "Cannot use both --documentation-only and --interpretation-only together"
            )
            sys.exit(1)

        # Embed each model
        for model in selected_models:
            if verbose:
                logger.debug(f"Processing model {model.name}")

            # Store documentation embedding if requested
            if not interpretation_only:
                model_text = model.get_readable_representation()

                # Create metadata
                metadata = {
                    "schema": model.schema,
                    "materialization": model.materialization,
                }
                if hasattr(model, "tags") and model.tags:
                    metadata["tags"] = model.tags

                logger.info(f"Storing documentation embedding for model {model.name}")
                vector_store.store_model(model.name, model_text, metadata)

            # Store interpretation embedding if requested and available
            if not documentation_only and model.interpretation:
                logger.info(f"Storing interpretation embedding for model {model.name}")
                vector_store.store_model_interpretation(
                    model.name, model.interpretation
                )
            elif not documentation_only and verbose:
                logger.debug(
                    f"Skipping interpretation embedding for model {model.name} - no interpretation available"
                )

        logger.info(f"Successfully embedded {len(selected_models)} models")
        return 0

    except Exception as e:
        logger.error(f"Error embedding models: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option(
    "--postgres-connection-string",
    help="PostgreSQL connection string for vector database",
    envvar="POSTGRES_CONNECTION_STRING",
)
@click.option("--openai-api-key", help="OpenAI API key", envvar="OPENAI_API_KEY")
@click.option("--openai-model", help="OpenAI model to use", envvar="OPENAI_MODEL")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--use-interpretation",
    is_flag=True,
    help="Use model interpretations for search instead of documentation",
)
def ask(
    question,
    postgres_uri,
    postgres_connection_string,
    openai_api_key,
    openai_model,
    verbose,
    use_interpretation,
):
    """
    Ask a question about your dbt project.

    By default, searches are performed using documentation embeddings.
    Use --use-interpretation to search using interpretation embeddings.
    """
    try:
        # Set logging level based on verbosity
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            logger.info("Loaded environment variables from .env file")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage
        from dbt_llm_agent.storage.vector_store import PostgresVectorStore
        from dbt_llm_agent.storage.question_service import QuestionTrackingService
        from dbt_llm_agent.core.agent import DBTAgent

        # Get PostgreSQL URI
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Get PostgreSQL connection string for vector database
        if not postgres_connection_string:
            postgres_connection_string = get_env_var("POSTGRES_CONNECTION_STRING")
            if not postgres_connection_string:
                postgres_connection_string = postgres_uri
                logger.info("Using POSTGRES_URI for vector database connection string")

        # Get OpenAI API key
        if not openai_api_key:
            openai_api_key = get_env_var("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error(
                    "OpenAI API key not provided. Please either:\n"
                    "1. Add OPENAI_API_KEY to your .env file\n"
                    "2. Pass it as --openai-api-key argument"
                )
                sys.exit(1)

        # Get OpenAI model
        if not openai_model:
            openai_model = get_env_var("OPENAI_MODEL", "gpt-4-turbo")

        # Initialize storage
        logger.info(f"Connecting to PostgreSQL database: {postgres_uri}")
        postgres = PostgresStorage(postgres_uri)

        # Initialize vector store
        logger.info(f"Connecting to vector database: {postgres_connection_string}")
        vector_store = PostgresVectorStore(connection_string=postgres_connection_string)

        # Initialize question tracking
        question_tracking = QuestionTrackingService(postgres_uri)

        # Initialize agent
        logger.info(f"Initializing DBT agent with {openai_model} model")
        agent = DBTAgent(
            postgres_storage=postgres,
            vector_store=vector_store,
            openai_api_key=openai_api_key,
            model_name=openai_model,
        )

        # Indicate which embedding type is being used
        if use_interpretation:
            logger.info("Using interpretation embeddings for search")
        else:
            logger.info("Using documentation embeddings for search")

        # Ask the question
        logger.info(f"Asking: {question}")
        result = agent.answer_question(question, use_interpretation=use_interpretation)

        # Output the answer
        colored_echo("\nAnswer:", color="INFO", bold=True)
        colored_echo(result["answer"])

        # List the models used
        if result["relevant_models"]:
            colored_echo("\nModels used:", color="INFO", bold=True)
            for model_data in result["relevant_models"]:
                # Show which models used interpretation
                model_info = f"- {model_data['name']}"
                if (
                    "used_interpretation" in model_data
                    and model_data["used_interpretation"]
                ):
                    model_info += " (used interpretation)"
                colored_echo(model_info, color="DEBUG")

        # Record the question and answer
        model_names = [model["name"] for model in result["relevant_models"]]
        question_id = question_tracking.record_question(
            question_text=question,
            answer_text=result["answer"],
            model_names=model_names,
        )

        colored_echo(f"\nQuestion ID: {question_id}", color="INFO", bold=True)
        colored_echo("You can provide feedback on this answer with:", color="INFO")
        colored_echo(
            f"  dbt-llm-agent feedback {question_id} --useful=true", color="DEBUG"
        )

        return 0

    except Exception as e:
        logger.error(f"Error asking question: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("question_id", type=int)
@click.option("--useful", type=bool, required=True, help="Was the answer useful?")
@click.option("--feedback", help="Additional feedback")
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
def feedback(question_id, useful, feedback, postgres_uri):
    """
    Provide feedback on an answer.
    """
    try:
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.question_service import QuestionTrackingService

        # Get PostgreSQL URI
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Initialize question tracking
        question_tracking = QuestionTrackingService(postgres_uri)

        # Get the question to make sure it exists
        question = question_tracking.get_question(question_id)
        if not question:
            logger.error(f"Question with ID {question_id} not found")
            sys.exit(1)

        # Update feedback
        success = question_tracking.update_feedback(
            question_id=question_id, was_useful=useful, feedback=feedback
        )

        if success:
            colored_echo(
                f"Feedback recorded for question {question_id}", color="INFO", bold=True
            )
        else:
            colored_echo(
                f"Failed to record feedback for question {question_id}", color="ERROR"
            )

        return 0

    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        sys.exit(1)


@cli.command()
@click.option("--limit", type=int, default=10, help="Number of questions to show")
@click.option("--offset", type=int, default=0, help="Offset for pagination")
@click.option("--useful", type=bool, help="Filter by usefulness")
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
def questions(limit, offset, useful, postgres_uri):
    """
    List questions and answers.
    """
    try:
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.question_service import QuestionTrackingService

        # Get PostgreSQL URI
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Initialize question tracking
        question_tracking = QuestionTrackingService(postgres_uri)

        # Get questions
        questions = question_tracking.get_all_questions(
            limit=limit, offset=offset, was_useful=useful
        )

        if not questions:
            colored_echo("No questions found", color="WARNING")
            return 0

        colored_echo(f"Found {len(questions)} questions:", color="INFO", bold=True)
        for q in questions:
            colored_echo(f"\nID: {q['id']}", color="INFO", bold=True)
            colored_echo(f"Question: {q['question_text']}", color="INFO")
            colored_echo(f"Answer: {q['answer_text'][:100]}...", color="DEBUG")
            # Use different colors based on usefulness
            usefulness_color = "INFO" if q["was_useful"] else "WARNING"
            colored_echo(f"Was useful: {q['was_useful']}", color=usefulness_color)
            colored_echo(f"Models: {', '.join(q['models'])}", color="DEBUG")
            colored_echo(f"Created at: {q['created_at']}", color="DEBUG")

        return 0

    except Exception as e:
        logger.error(f"Error listing questions: {e}")
        sys.exit(1)


@cli.command()
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--revision", help="Target revision (default: head)", default="head")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def migrate(postgres_uri, revision, verbose):
    """Update the database schema to the latest version.

    This command applies Alembic migrations to update the database schema.

    You can specify a specific revision with --revision (default: head).
    """
    set_logging_level(verbose)

    # Load configuration if not provided
    if not postgres_uri:
        postgres_uri = get_config_value("postgres_uri")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided and not found in config")
        sys.exit(1)

    try:
        logger.info("Running database migrations...")

        # Initialize PostgresStorage and apply migrations explicitly
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage

        postgres_storage = PostgresStorage(postgres_uri)

        # Apply migrations using the storage class method
        success = postgres_storage.apply_migrations()

        if success:
            logger.info("Migrations completed successfully")
        else:
            logger.error("Migration failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def init_db(postgres_uri, verbose):
    """Initialize the database schema.

    This command creates all tables and initializes the database with the latest schema.
    """
    set_logging_level(verbose)

    # Import necessary modules
    import sqlalchemy as sa
    from dbt_llm_agent.storage.models import Base
    from dbt_llm_agent.storage.postgres_storage import PostgresStorage

    # Load configuration if not provided
    if not postgres_uri:
        postgres_uri = get_config_value("postgres_uri")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided and not found in config")
        sys.exit(1)

    try:
        logger.info("Initializing database schema...")

        # Create the storage instance
        postgres_storage = PostgresStorage(postgres_uri)

        # Apply migrations explicitly
        success = postgres_storage.apply_migrations()

        if success:
            logger.info("Database initialization completed successfully")
        else:
            logger.error("Database initialization failed during migration step")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option(
    "--select",
    required=True,
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def list(select, postgres_uri, verbose):
    """List selected models from the database."""
    set_logging_level(verbose)

    if not postgres_uri:
        postgres_uri = get_config_value("postgres_uri")

    if not postgres_uri:
        colored_echo(
            "PostgreSQL URI not provided and not found in config",
            color="RED",
            bold=True,
        )
        sys.exit(1)

    try:
        # Initialize PostgreSQL storage
        postgres_storage = PostgresStorage(postgres_uri)

        # Fetch all models from the database
        all_models = postgres_storage.get_all_models()

        if not all_models:
            colored_echo("No models found in the database", color="YELLOW")
            return

        # Select models based on the provided selection
        selector = ModelSelector(all_models)
        selected_models = selector.select_models(select)

        if not selected_models:
            colored_echo(
                f"No models selected using '{select}'", color="YELLOW", bold=True
            )
            return

        colored_echo(
            f"Selected {len(selected_models)} model(s) using '{select}':",
            color="GREEN",
            bold=True,
        )

        for idx, model in enumerate(selected_models, 1):
            colored_echo(
                f"{idx}. {model.name} ({model.materialization}, {model.schema})"
            )
            if verbose:
                if model.description:
                    colored_echo(f"   Description: {model.description}", color="CYAN")
                colored_echo(f"   Path: {model.path}", color="CYAN")
                if model.columns:
                    colored_echo(f"   Columns:", color="CYAN")
                    for col_name, col in model.columns.items():
                        desc = f" - {col.description}" if col.description else ""
                        colored_echo(f"     - {col_name}{desc}", color="CYAN")
                colored_echo("")

    except Exception as e:
        colored_echo(f"Error listing models: {str(e)}", color="RED", bold=True)
        if verbose:
            import traceback

            colored_echo(traceback.format_exc(), color="RED")
        sys.exit(1)


@cli.command()
@click.argument("model_name", required=True)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option(
    "--postgres-connection-string",
    help="PostgreSQL connection string for vector database",
    envvar="POSTGRES_CONNECTION_STRING",
)
@click.option("--openai-api-key", help="OpenAI API key", envvar="OPENAI_API_KEY")
@click.option("--openai-model", help="OpenAI model to use", envvar="OPENAI_MODEL")
@click.option(
    "--no-save", is_flag=True, help="Don't save the documentation to the database"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--embed", is_flag=True, help="Embed the interpretation in the vector store"
)
def interpret(
    model_name,
    postgres_uri,
    postgres_connection_string,
    openai_api_key,
    openai_model,
    no_save,
    verbose,
    embed,
):
    """Interpret a model and generate documentation for it.

    This command analyzes the SQL code of a model along with its upstream dependencies
    to generate documentation in YAML format.

    By default, the documentation is saved to the database. Use --no-save to disable this.
    """
    set_logging_level(verbose)

    # Import necessary modules
    from dbt_llm_agent.storage.postgres_storage import PostgresStorage
    from dbt_llm_agent.storage.vector_store import PostgresVectorStore
    from dbt_llm_agent.core.agent import DBTAgent
    from dbt_llm_agent.utils.model_selector import ModelSelector

    # Load configuration and override with command line arguments
    if not postgres_uri:
        postgres_uri = get_config_value("postgres_uri")

    if not postgres_connection_string:
        postgres_connection_string = get_config_value("postgres_connection_string")
        if not postgres_connection_string and postgres_uri:
            # Use postgres_uri as a fallback
            postgres_connection_string = postgres_uri

    if not openai_api_key:
        openai_api_key = get_config_value("openai_api_key")

    if not openai_model:
        openai_model = get_config_value("openai_model")
        if not openai_model:
            openai_model = "gpt-4-turbo"

    # Validate configuration
    if not postgres_uri:
        logger.error("PostgreSQL URI not provided and not found in config")
        sys.exit(1)

    if not postgres_connection_string:
        logger.error(
            "PostgreSQL connection string not provided and not found in config"
        )
        sys.exit(1)

    if not openai_api_key:
        logger.error("OpenAI API key not provided and not found in config")
        sys.exit(1)

    try:
        logger.info(f"Interpreting model: {model_name}")

        # Initialize storage and agent
        postgres_storage = PostgresStorage(postgres_uri)
        vector_store = PostgresVectorStore(postgres_connection_string)
        agent = DBTAgent(
            postgres_storage=postgres_storage,
            vector_store=vector_store,
            openai_api_key=openai_api_key,
            model_name=openai_model,
        )

        # Check if the model exists
        model = postgres_storage.get_model(model_name)
        if not model:
            logger.error(f"Model '{model_name}' not found in the database")
            sys.exit(1)

        # Interpret the model
        result = agent.interpret_model(model_name)

        if "error" in result:
            logger.error(f"Error interpreting model: {result['error']}")
            sys.exit(1)

        # If verbose, print the prompt that was used
        if verbose and "prompt" in result:
            logger.info("Prompt used for interpretation:")
            print(result["prompt"])

        # Print the resulting YAML documentation
        if verbose:
            logger.info("Generated YAML Documentation:")
        print(result["yaml_documentation"])

        # Save the documentation by default unless --no-save is provided
        if not no_save:
            logger.info("Saving documentation to database...")
            save_result = agent.save_interpreted_documentation(
                model_name, result["yaml_documentation"]
            )

            if save_result["success"]:
                logger.info(f"Documentation saved successfully for model: {model_name}")

                # Embed the interpretation if requested
                if embed:
                    logger.info("Embedding interpretation in vector store...")
                    # Get the updated model with the interpretation
                    updated_model = postgres_storage.get_model(model_name)
                    if updated_model and updated_model.interpretation:
                        vector_store.store_model_interpretation(
                            model_name, updated_model.interpretation
                        )
                        logger.info(
                            f"Interpretation embedded successfully for model: {model_name}"
                        )
                    else:
                        logger.error("Could not find interpretation to embed")
            else:
                error_msg = save_result.get("error", "Unknown error")
                logger.error(f"Failed to save documentation: {error_msg}")

    except Exception as e:
        logger.error(f"Error interpreting model: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)


def main():
    # Set up colored logging - only configure once
    if not logging.root.handlers:
        setup_logging()

    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
        logger.debug("Loaded environment variables from .env file")
    except ImportError:
        logger.warning(
            "python-dotenv not installed. Environment variables may not be properly loaded."
        )

    cli()


if __name__ == "__main__":
    main()
