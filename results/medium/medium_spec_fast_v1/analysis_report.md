# Medium results report

This Medium run implements TWO enhancements:
1) CNN-VAE on log-mel spectrograms (audio branch)
2) Hybrid audio+lyrics via a shared latent representation

Clustering: KMeans, Agglomerative, DBSCAN.

Labels were available, so ARI is included.

## Best configuration per representation (by silhouette, tie-breaker: lowest DB)
| repr | algo | k | eps | min_samples | silhouette | davies_bouldin | ari |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_spec_mmvae | dbscan | nan | 0.3 | 10.0 | 0.0711 | 1.0943 | 0.0124 |
| audio_pca_baseline | kmeans6 | 6.0 | nan | nan | 0.2418 | 1.5109 | 0.0572 |
| lyrics_baseline | dbscan | nan | 1.1 | 20.0 | 0.1277 | 2.3865 | 0.0088 |
| hybrid_pca_baseline | kmeans6 | 6.0 | nan | nan | 0.1954 | 1.7020 | 0.0593 |

## Why VAE may be better/worse than baselines
- Better: the CNN encoder can learn non-linear time-frequency structure and denoise nuisance variation, making the latent more clusterable than simple PCA baselines.
- Worse: VAE training optimizes reconstruction + KL, which may not align with cluster separation; the latent can be over-smoothed or collapse toward a small number of modes.
- Hybrid-specific: if lyrics embeddings carry weak/orthogonal signal or are imbalanced vs audio, the shared latent can under-utilize lyrics unless weighting/architecture is tuned.
