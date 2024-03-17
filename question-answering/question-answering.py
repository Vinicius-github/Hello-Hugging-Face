from transformers import pipeline

#O pipeline question-answering responde perguntas usando informações dado um contexto:
#Note que o pipeline funciona através da extração da informação dado um contexto; não gera uma resposta.

question_answerer = pipeline("question-answering")
res = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(res)
