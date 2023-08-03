"""Search through documents using LangChain and LlamaIndex."""

from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    VectorStoreIndex,
)

from llama_index.chat_engine.types import ChatMode
from langchain.chat_models import ChatOpenAI

data_folder_path = "data"

llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0613")
llm_predictor = LLMPredictor(llm=llm)

max_input_size = 4096
num_output = 256
max_chunk_overlap_ratio = 0.2

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap_ratio)

# Load documents from the 'data' directory
documents = SimpleDirectoryReader(data_folder_path).load_data()
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper
)
index: VectorStoreIndex = GPTVectorStoreIndex.from_documents(
    documents, service_context=service_context, show_progress=True
)

chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT)

chat_engine.chat_repl()
