from transformers import pipeline

## Por padrão, esse pipeline seleciona particularmente um modelo pré-treinado que tem sido ajustado (fine-tuned) para análise 
## de sentimentos em Inglês. O modelo é baixado e cacheado quando você cria o objeto classifier. Se você rodar novamente o comando, 
## o modelo cacheado irá ser usado no lugar e não haverá necessidade de baixar o modelo novamente.

classifier = pipeline("sentiment-analysis")
res = classifier("I've been waiting for a Hugging Face course my whole life!")
print(res)

res2 = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)

print(res2)
