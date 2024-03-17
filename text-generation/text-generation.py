from transformers import pipeline

#Geração de Texto
#A principal ideia aqui é que você coloque um pedaço de texto 
#e o modelo irá autocompletá-lo ao gerar o texto restante. Isso é similar ao recurso de predição textual que é encontrado em inúmeros 
#celulares. A geração de texto envolve aleatoriedade, então é normal se você não obter o mesmo resultado obtido rodando o modelo várias vezes.

generator = pipeline("text-generation")
res = generator(
    "In this course, we will teach you how to"
) 
print(res)


# Quando você não especifica qual modelo utilizar no pipeline, ele utiliza o modelo padrão do tópico escolhido.
# Podemos escolher diferentes modelos no Model Hub do Hugging Face e você pode configurar parametros para o modelo.

generator2 = pipeline("text-generation", model="distilgpt2")
res2 = generator2(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(res2)
