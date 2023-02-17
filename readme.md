# Distant Supervised Movie Title Extraction Using FLAN-T5

Generative movie title retrieval using FLAN-T5 in 🤗HuggingFace ecosystem.

### TL;DR.

Turned out you can bootstrap a reasonably performing movie title (generative) NER model from GPT-3! Our FLAN-T5 model tuned on davinci-003's noisy supervision outperforms its teacher, and has strong training-free performance on MIT Movies Corpus. 

### Why This model, and How

We were looking for a system that could extract movie titles from [noisy user-generated text on reddit](https://www.reddit.com/r/MovieSuggestions/) but could only find [MLMs](https://nlp.johnsnowlabs.com/2021/07/20/ner_mit_movie_simple_distilbert_base_cased_en.html) fine-tuned on MIT Movie Corpus. However, the MIT Movie Corpus has limited type of movie titles, and there is usually at most 1 movie per post, thus models tuned on it does not transfer well to reddit posts. 

We ended up using GPT-3 model to generate distant labels. In particular, we first prompt GPT-3 for movie titles in reddit posts, then fine tune a Flan-T5 model on GPT-3's outputs using standard language modeling loss. We evaluate the model on 400 turker annotated samples released repeated for three batches.

### Evaluations

#### In domain

We ask turkers to extract movie titles by copy-pasting movie names from the posts, separated by comma. However, in practice the inter-annotator agreement is low.  On average the annotators only fully agree with each other on about 200 out of 400 samples, and only about 300 times two of the three annotators agree.

Never the less, averaging across three batches, the model should land somewhere above 0.75 towards 0.80 F/P/R scores. In the meantime, we consider a title mentioned by any of the 3 turkers per sample a correct retrieval, our precision score is at 85%.

#### Domain transfer to MIT Movies

The model is mostly intended for reddit posts, but works slightly out of domain as well. We achieve 0.79 F score on MIT Movie Trivia out of the box, which is close to the current supervised model's performance of 0.84. This could be due to the type of signals are different across datasets: in reddit posts, movie name usually appears in bullet lists/are separated by comma, where as in MIT movie dataset it is usally explicitly mentioned that a title or a movie is appearing.

### Limitations and Technical Stuff

Comments that are marked '[deleted]' and '[removed]' are ignored in data curation, if undefined model behavior pops up please consider heuristic based filtering.

See ```gpt_queries.py``` for how we prompted the davinci-003 checkpoint.





