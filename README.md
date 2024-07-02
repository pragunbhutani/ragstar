# dbt-llm-tools aka. ragstar

**LLM-based tools for dbt projects**

dbt-llm-tools, also known as ragstar, provides a suite of tools powered by Large Language Models (LLMs) to enhance your dbt project workflow. It allows you to ask questions about your data and generate documentation for your models.

**Here is a quick demo of how the Chatbot works:**

https://www.loom.com/share/abb0612c4e884d4cb8fabc22af964e7e?sid=f5f8c0e6-51f5-4afc-a7bf-51e9e182c2e7

### Key functionalities

* **Chatbot:** Ask questions about your data directly using the chatbot. It leverages your dbt model documentation to provide insightful answers.
* **Documentation Generator:** Generate comprehensive documentation for your dbt models, including descriptions and lineage information.


### Getting Started

To install `dbt-llm-tools` with the UI:

1. Clone the repository:
   ```bash
   gh repo clone pragunbhutani/dbt-llm-tools
   ```
2. Navigate to the project directory:
   ```bash
   cd dbt-llm-tools
   ```
3. Install Poetry:
   ```bash
   make poetry
   ```
   - Add the poetry executable to your PATH and refresh the terminal.
4. Install the project dependencies:
   ```bash
   make install
   ```
5. Install an example project (optional):
   ```bash
   make fetch_example_project
   ```
6. Run the UI:
   ```bash
   make run_client
   ```
7. Optional Step (if any dependency error shows up) :
   ```bash
   pip install dbt-llm-tools
   ```

This will launch the client in your browser at `http://localhost:8501/app`.

**Note:** An OpenAI API key is required to use the tools.

### Documentation

For detailed instructions and API reference, refer to the official documentation: [https://dbt-llm-tools.readthedocs.io/en/latest/](https://dbt-llm-tools.readthedocs.io/en/latest/)

### Classes

* **Chatbot:**
  - Loads your dbt project information and creates a local vector store.
  - Allows you to ask questions about your data.
  - Retrieves relevant models and utilizes ChatGPT to generate responses.
  - Currently supports OpenAI ChatGPT models.

```python
from dbt_llm_tools import Chatbot

# Instantiate a chatbot object
chatbot = Chatbot(
	dbt_project_root='/path/to/dbt/project',
	openai_api_key='YOUR_OPENAI_API_KEY',
)

# Step 1. Load models information from your dbt ymls into a local vector store
chatbot.load_models()

# Step 2. Ask the chatbot a question
response = chatbot.ask_question(
	'How can I obtain the number of customers who upgraded to a paid plan in the last 3 months?'
)
print(response)
```

* **Documentation Generator:**
  - Generates documentation for your dbt models and their dependencies.
  - Requires your OpenAI API key.

```python
from dbt_llm_tools import DocumentationGenerator

# Instantiate a Documentation Generator object
doc_gen = DocumentationGenerator(
	dbt_project_root="YOUR_DBT_PROJECT_PATH",
	openai_api_key="YOUR_OPENAI_API_KEY",
)

# Generate documentation for a model and all its upstream models
doc_gen.generate_documentation(
	model_name='dbt_model_name',
	write_documentation_to_yaml=False
)
```

#### How it works

The Chatbot is based on the concept of Retrieval Augmented Generation and basically works as follows:

- When you call the `chatbot.load_models()` method, the bot scans all the folders in the locations specified by you for dbt YML files.
- It then converts all the models into a text description, which are stored as embeddings in a vector database. The bot currently only supports [ChromaDB](https://www.trychroma.com/) as a vector db, which is persisted in a file on your local machine.
- When you ask a query, it fetches 3 models whose description is found to be the most relevant for your query.
- These models are then fed into ChatGPT as a prompt, along with some basic instructions and your question.
- The response is returned to you as a string.

## Partners

* [JIIT's Open Source Developers Community](https://github.com/osdc)
