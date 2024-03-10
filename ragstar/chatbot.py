from openai import OpenAI

from ragstar.types import PromptMessage, ParsedSearchResult

from ragstar.dbt_project import DbtProject
from ragstar.vector_store import VectorStore


class Chatbot:
    """
    A class representing a chatbot that allows users to ask questions about dbt models.

    Attributes:
        project (DbtProject): The dbt project being used by the chatbot.
        store (VectorStore): The vector store being used by the chatbot.

    Methods:
        set_embedding_model: Set the embedding model for the vector store.
        set_chatbot_model: Set the chatbot model for the chatbot.
        get_instructions: Get the instructions for the chatbot.
        set_instructions: Set the instructions for the chatbot.
        load_models: Load the models into the vector store.
        reset_model_db: Reset the model vector store.
        ask_question: Ask the chatbot a question and get a response.
    """

    def __init__(
        self,
        dbt_project_root: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-large",
        chatbot_model: str = "gpt-4-turbo-preview",
        db_persist_path: str = "./chroma.db",
    ) -> None:
        """
        Initializes a chatbot object along with a default set of instructions.

        Args:
            dbt_project_root (str): The absolute path to the root of the dbt project.
            openai_api_key (str): Your OpenAI API key.

            embedding_model (str, optional): The name of the OpenAI embedding model to be used.
            Defaults to "text-embedding-3-large".

            chatbot_model (str, optional): The name of the OpenAI chatbot model to be used.
            Defaults to "gpt-4-turbo-preview".

            db_persist_path (str, optional): The path to the persistent database file. Defaults to "./chroma.db".

        Returns:
            None
        """
        self.__chatbot_model: str = chatbot_model
        self.__openai_api_key: str = openai_api_key

        self.project: DbtProject = DbtProject(dbt_project_root)
        self.store: VectorStore = VectorStore(
            openai_api_key, embedding_model, db_persist_path
        )

        self.__instructions: list[str] = ["""
You are a data analyst working with a data warehouse.
You should provide the user with the information they need to answer their question.
You should only provide information that you are confident is correct.
When you are not sure about the answer, you should let the user know.
If you are able to construct a SQL query that would answer the user's question, you should do so.
However please refrain from doing so if the user's question is ambiguous or unclear.
When writing a SQL query, you should only use column values if these values have been explicitly provided to you in the information you have been given.
Do not write a SQL query if you are unsure about the correctness of the query or about the values contained in the columns.
Only write a SQL query if you are confident that the query is exhaustive and that it will return the correct results.
If it is not possible to write a SQL that fulfils these conditions, you should instead respond with the names of the tables or columns that you think are relevant to the user's question.
You should also refrain from providing any information that is not directly related to the user's question or that which cannot be inferred from the information you have been given.
The following information about tables and columns is available to you:
"""]

    def __prepare_prompt(
        self, closest_models: list[ParsedSearchResult], query: str
    ) -> list[PromptMessage]:
        """
        Prepare the prompt for the chatbot using instructions, closest models and the user query.

        Args:
            closest_models (list[ParsedSearchResult]): The closest models to the user query.
            query (str): The user query.

        Returns:
            list[PromptMessage]: A list of prompt messages to be used by the chatbot.
        """
        prompt: list[PromptMessage] = []

        for instruction in self.__instructions:
            prompt.append({"role": "system", "content": instruction})

        for model in closest_models:
            prompt.append({"role": "system", "content": model["document"]})

        prompt.append({"role": "user", "content": query})

        return prompt

    def set_embedding_model(self, model: str) -> None:
        """
        Set the embedding model for the vector store.

        Args:
            model (str): The name of the OpenAI embedding model to be used.

        Returns:
            None
        """
        self.store.set_embedding_fn(model)

    def set_chatbot_model(self, model: str) -> None:
        """
        Set the chatbot model for the chatbot.

        Args:
            model (str): The name of the OpenAI chatbot model to be used.

        Returns:
            None
        """
        self.__chatbot_model = model

    def get_instructions(self) -> list[str]:
        """
        Get the instructions being used to tune the chatbot.

        Returns:
            list[str]: A list of instructions being used to tune the chatbot.
        """
        return self.__instructions

    def set_instructions(self, instructions: list[str]) -> None:
        """
        Set the instructions for the chatbot.

        Args:
            instructions (list[str]): A list of instructions for the chatbot.

        Returns:
            None
        """
        self.__instructions = instructions

    def load_models(
        self,
        models: list[str] = None,
        included_folders: list[str] = None,
        excluded_folders: list[str] = None,
    ) -> None:
        """
        Upsert the set of models that will be available to your chatbot into a vector store.
        The chatbot will only be able to use these models to answer questions and nothing else.

        The default behavior is to load all models in the dbt project, but you can specify a subset of models,
        included folders or excluded folders to customize the set of models that will be available to the chatbot.

        Args:
            models (list[str], optional): A list of model names to load into the vector store.

            included_folders (list[str], optional): A list of paths to all folders that should be included
            in model search. Paths are relative to dbt project root.

            exclude_folders (list[str], optional): A list of paths to all folders that should be excluded
            in model search. Paths are relative to dbt project root.

        Returns:
            None
        """
        models = self.project.get_models(models, included_folders, excluded_folders)
        self.store.upsert_models(models)

    def reset_model_db(self) -> None:
        """
        This will reset and remove all the models from the vector store.
        You'll need to load the models again using the load_models method if you want to use the chatbot.

        Returns:
            None
        """
        self.store.reset_collection()

    def ask_question(self, query: str, get_models_name_only: bool = False) -> str:
        """
        Ask the chatbot a question about your dbt models and get a response.
        The chatbot looks the dbt models most similar to the user query and uses them to answer the question.

        Args:
            query (str): The question you want to ask the chatbot.

        Returns:
            str: The chatbot's response to your question.
        """
        print("Asking question: ", query)

        print("\nLooking for closest models to the query...")

        closest_models = self.store.query_collection(query)
        model_names = ", ".join(map(lambda x: x["id"], closest_models))

        if get_models_name_only:
            return model_names

        print("Closest models found:", model_names)

        print("\nPreparing prompt...")
        prompt = self.__prepare_prompt(closest_models, query)

        client = OpenAI(api_key=self.__openai_api_key)

        print("\nCalculating response...")
        completion = client.chat.completions.create(
            model=self.__chatbot_model,
            messages=prompt,
        )

        print("\nResponse received: \n")
        print(completion.choices[0].message.content)

        return completion.choices[0].message
