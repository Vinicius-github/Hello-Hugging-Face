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
