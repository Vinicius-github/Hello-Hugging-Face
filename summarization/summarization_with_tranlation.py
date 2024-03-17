#Após a sumarização do texto, faço uma comparação entra a tradução para português utilizando o google e o hugging face.

from transformers import pipeline
from googletrans import Translator


# Carregar o pipeline para sumarização de texto em inglês
summarization_pipeline = pipeline("summarization", model="t5-base", tokenizer="t5-base")

# Texto de exemplo em inglês
english_text = """ America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers."""

# Sumarizar o texto em inglês
summarized_text = summarization_pipeline(english_text)[0]["summary_text"]

# Traduzir o texto sumarizado para português usando a API do Google Translate
translator = Translator()
translated_text = translator.translate(summarized_text, src="en", dest="pt").text

# Traduzir o texto sumarizado para português usando a API da Microsoft Azure
translator_hug = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-pt")
translated_text_hug = translator_hug(summarized_text)

print("Texto Sumarizado em Inglês:", summarized_text)
print("Texto Traduzido para Português com API Google:", translated_text)
print("Texto Traduzido para Português com API HUG:", translated_text_hug)
