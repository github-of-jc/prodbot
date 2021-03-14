import nltk

print("----start----")
sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""

print("load sentence: %s" % sentence)
tokens = nltk.word_tokenize(sentence)
print("load tokens: %s" % tokens)


tagged = nltk.pos_tag(tokens)
print("load tagged: %s" % tagged)


print(tagged[0:6])

print("--entities--")
entities = nltk.chunk.ne_chunk(tagged)

print(entities)