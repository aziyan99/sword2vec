from sword2vec import SkipGramWord2Vec

model = SkipGramWord2Vec(
    window_size=2,
    learning_rate=0.025,
    embedding_dim=2,
    epochs=5,
    stopword_file="./id_stopword.txt",
)

lines = []
with open("./datasets_judul_buku_full.txt", encoding="utf8") as f:
    for line in f.readlines():
        lines.append(line)
    f.close()

model.train(lines=lines)
model.save_compressed_model("model")

loaded_model = SkipGramWord2Vec.load_compressed_model("./model")

word_test = "belajar"

# use general predict
pred_words = loaded_model.predict(word=word_test, topn=20)

print("\npredict:\n")
for word in pred_words:
    print(word)

print("\ncosine similarity:\n")
cos_words = loaded_model.search_similar_words(word=word_test, topn=20)
for word in cos_words:
    print(word)
