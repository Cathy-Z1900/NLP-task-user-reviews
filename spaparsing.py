import spacy
from spacy import displacy

# Load the language model
nlp = spacy.load('\en_core_web_sm\en_core_web_sm-3.2.0')
sentence = 'sentence'

doc = nlp(sentence)

print("{:<15} | {:<8} | {:<15} | {:<20}".format('Token', 'Relation', 'Head', 'Children'))
print("-" * 70)

for token in doc:
    print("{:<15} | {:<8} | {:<15} | {:<20}"
          .format(str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])))
displacy.render(doc, style='dep', jupyter=True, options={'distance': 120})