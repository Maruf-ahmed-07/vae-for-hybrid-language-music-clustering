# Hard results report

Hard model: Conditional VAE (CVAE) conditioned on genre.

Modalities: log-mel spectrogram (Conv2D) + lyrics embedding (TF-IDF+SVD).

Evaluation metrics: Silhouette, Davies–Bouldin, ARI, NMI, Purity.

## Best by ARI (per representation)

| repr | algo | k | silhouette | davies_bouldin | ari | nmi | purity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cvae_latent | kmeans8 | 8 | 0.4156855642795563 | 1.1814928957021267 | 0.13733969713466251 | 0.09797512417721498 | 0.6931780366056572 |
| pca_concat | kmeans6 | 6 | 0.1953829973936081 | 1.7020171044227876 | 0.05929445213651615 | 0.21184340990881606 | 0.6931780366056572 |
| ae_concat | kmeans6 | 6 | 0.08582001179456711 | 2.6570307742920467 | 0.06309626156381173 | 0.20781965731097377 | 0.6945091514143095 |
| direct_spectral | kmeans6 | 6 | 0.22665905952453613 | 1.6082834990941421 | 0.0606065856549788 | 0.2110375216693062 | 0.6931780366056572 |


## Notes
- CVAE uses genre as conditioning variable c during encoding/decoding.
- Language distribution plots are omitted (no reliable language labels).
