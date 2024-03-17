from transformers import pipeline

## O pipeline zero-shot-classification permite você especificar quais rótulos usar 
## para a classificação que você pode configurar. Desse modo você não precisa “confiar” nos rótulos dos modelos pré-treinados. 

classifier = pipeline("zero-shot-classification")
res = classifier2(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(res)
