# DeBERTa for Othello

 DeBERTa is a language model. For it to understand Othello, we prepared the word dictionary ["[PAD]","[CLS]","BLACK","SPACE","WHITE"].

 It contains five words, so very memory efficient. We also limit the sentence length to 37. It contains "CLS" and the information of squares of 6x6 Othello.

 Thanks to the optimizations, the size of DeBERTa became 43 MB. We used transformers DeBERTa V2 model with the "x-small" configulation of DeBERTa V3. It's about 400 MB originally, and became 10 times smaller by the optimizations.

 DeBERTa has relative position embeddings, but we thought absolute position embeddings are also needed, so we added it. We haven't tried this without it, so the contribution of it is uncertain.

 Result: The DeBERTa model, after 3 hours of training, exhibited performance equivalent to a [CNN model](https://github.com/dochy-ksti/rust_general_alphazero_othello) trained for 43 minutes.

 If you want to try this yourself, "cargo build --release" and "python -m python.main release". This requires pytorch and CUDA.
