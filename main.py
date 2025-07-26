from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import retriever

model = OllamaLLM(model='llama3.2')

template = """ 
You are a helpful assistant who specializes in providing restaurant reviews and suggestions.
here are some relevant reviews: {reviews}
here is the question that you are supposed to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


while True:
    print ("_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*")

    question = input("Ask me a Question! (press q to quit)")
    if question == 'q':
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke(
        {
            "reviews": [],
            "question": question
        }
    )

    print(result)