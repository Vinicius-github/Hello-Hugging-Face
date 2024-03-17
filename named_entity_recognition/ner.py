from transformers import pipeline


#Reconhecimento de Entidades Nomeadas (NER) é uma tarefa onde o modelo tem de achar quais partes do texto correspondem a entidades como pessoas, locais, organizações.

ner = pipeline("ner", grouped_entities=True)
res = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(res)

#Foi informado a opção grouped_entities=True na criação da função do pipeline para dize-lo para reagrupar juntos as partes da 
#sentença que correspondem à mesma entidade: aqui o modelo agrupou corretamente “Hugging” e “Face” como única organização, 
#ainda que o mesmo nome consista em múltiplas palavras
