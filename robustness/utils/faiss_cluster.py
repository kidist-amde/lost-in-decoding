import faiss
import numpy as np

def cluster_embeddings_with_faiss(embeddings, n_clusters, seed=2026, niter=300, use_gpu=False, verbose=True, score_function='cos_sim'):
    """
    Use FAISS to cluster embeddings using K-means   

    Args:
        embeddings: numpy array-like, shape (n_samples, embedding_dim)
        n_clusters: int, number of clusters (must be 1..n_samples)
        seed: int, random seed
        niter: int, K-means iteration times
        use_gpu: bool, whether to use GPU (if GPU/library is not available, it will automatically fall back to CPU)
        verbose: bool, whether to print detailed information
        score_function: according to score_function, select whether to normalize the embeddings using L2;
                        'cos_sim' means using cosine similarity (requires L2 normalization); other values do not normalize

    Returns:
        dict: containing the following key-value pairs
            - cluster_centers: numpy array, shape (n_clusters, embedding_dim), cluster centers
            - cluster_assignments: numpy array, shape (n_samples,), cluster assignments for each sample
            - cluster_stats: dict, statistics for each cluster
    """
    # 0. Basic input checks and preparation
    embeddings_np = np.asarray(embeddings)
    if embeddings_np.ndim != 2:
        raise ValueError(f"embeddings must be a two-dimensional array, current dimension is {embeddings_np.ndim}")
    n_samples, dim = embeddings_np.shape
    if n_samples == 0 or dim == 0:
        raise ValueError("embeddings shape is invalid: contains zero dimension")
    if not isinstance(n_clusters, (int, np.integer)):
        raise ValueError("n_clusters must be an integer")
    if n_clusters < 1 or n_clusters > n_samples:
        raise ValueError(f"n_clusters must be in the range [1, {n_samples}], current is {n_clusters}")

    normalization_enabled = (score_function == 'cos_sim')

    if verbose:
        print(f"Starting K-means clustering with {n_clusters} clusters on {n_samples} samples, dim={dim}...")
        print(f"Normalization: {'Enabled (cosine similarity)' if normalization_enabled else 'Disabled'}")
        print(f"Requested device: {'GPU' if use_gpu else 'CPU'}")

    # 1. Prepare embeddings
    embeddings_processed = embeddings_np.astype('float32', copy=True)

    # 2. Normalize L2 according to need
    if normalization_enabled:
        faiss.normalize_L2(embeddings_processed)

    # 3. Get embedding dimension
    d = embeddings_processed.shape[1]

    # 4. Initialize FAISS K-means
    def _build_kmeans(use_gpu_flag: bool):
        return faiss.Kmeans(
            d=d,
            k=n_clusters,
            niter=niter,
            verbose=verbose,
            gpu=use_gpu_flag,
            seed=seed
        )

    # Try to use the user-specified device first, if failed, automatically fall back to CPU
    try:
        kmeans = _build_kmeans(use_gpu)
    except Exception as e:
        if use_gpu:
            if verbose:
                print(f"[Warning] Failed to initialize GPU K-means, automatically fall back to CPU. Reason: {type(e).__name__}: {e}")
            kmeans = _build_kmeans(False)
            use_gpu = False
        else:
            raise

    # 5. Train K-means
    try:
        kmeans.train(embeddings_processed)
    except Exception as e:
        # Some environments expose the problem of GPU unavailability only after initialization and training
        if use_gpu:
            if verbose:
                print(f"[Warning] GPU training failed, try to train on CPU. Reason: {type(e).__name__}: {e}")
            kmeans = _build_kmeans(False)
            kmeans.train(embeddings_processed)
            use_gpu = False
        else:
            raise

    # 6. Get cluster centers
    cluster_centers = kmeans.centroids  # shape: (n_clusters, d)
    if verbose:
        print(f"Cluster centers shape: {cluster_centers.shape}")

    # 7. Get the cluster to which each sample belongs
    _, cluster_assignments = kmeans.index.search(embeddings_processed, 1)
    cluster_assignments = cluster_assignments.flatten()

    # 8. Calculate cluster statistics
    cluster_stats = {}
    if verbose:
        print("Clustering statistics:")
    for i in range(n_clusters):
        count = np.sum(cluster_assignments == i)
        cluster_stats[i] = int(count)
        if verbose:
            print(f"  Cluster {i}: {count} samples")

    return {
        "cluster_centers": cluster_centers,
        "cluster_assignments": cluster_assignments,
        "cluster_stats": cluster_stats
    }