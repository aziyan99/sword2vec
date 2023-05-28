from sword2vec import (
    SkipGramWord2Vec,
    Preprocessor,
    Tokenizer,
    StopWordRemover,
    LowerCaser,
)

preprocessor = (
    Preprocessor()
    .set_tokenizer(Tokenizer())
    .set_stop_word_remover(StopWordRemover())
    .set_lower_caser(LowerCaser())
)

model = SkipGramWord2Vec(
    window_size=5,
    learning_rate=0.025,
    embedding_dim=100,
    epochs=5,
    stopword_file="./id_stopword.txt",
)

# lines = []
# with open("./datasets_judul_buku_full.txt", encoding="utf8") as f:
#     for line in f.readlines():
#         lines.append(line)
#     f.close()

# model.train(lines=lines)
# model.save_compressed_model("model_epoch5_dim100_wsize5")

loaded_model = SkipGramWord2Vec.load_compressed_model("./model_epoch5_dim100_wsize5")

word_test = "asdasd"

tokens = preprocessor.preprocess(word_test)

# use general predict
# pred_words = loaded_model.predict(word=word_test, topn=20)

# print("\npredict:\n")
# for word in pred_words:
#     print(word)

print("\ncosine similarity:\n")
cos_words = loaded_model.search_similar_words(word=word_test, topn=20)
for word in cos_words:
    print(word)
