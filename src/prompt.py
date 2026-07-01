from langchain_core.prompts import PromptTemplate

# Your prompt text here
prompt_template = """
You are an expert AI assistant. Use ONLY the provided context to answer the question. Response to follow: keep responses short, clear,precise and easy to understand with source citations. If the answer is not present in the context, reply with: "I couldn't find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

# Create a PromptTemplate instance (optional, but good practice)
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# This variable is what your app.py needs
chain_type_kwargs = {"prompt": PROMPT}
