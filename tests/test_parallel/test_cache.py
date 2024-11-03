from rdkit.Chem import MolFromSmiles
from posebusters.parallel.cache import ComputationCache


def test_cache_initialization():
    """Test cache initialization."""
    cache = ComputationCache(max_size=100)
    assert cache.max_size == 100
    assert len(cache.cache) == 0


def test_cache_key_generation():
    """Test cache key generation for molecules."""
    cache = ComputationCache()
    mol = MolFromSmiles("CC")

    key = cache.get_key(mol, "test_computation")
    assert key is not None
    assert "test_computation" in key


def test_cache_storage_and_retrieval():
    """Test storing and retrieving results from cache."""
    cache = ComputationCache(max_size=2)

    # Store results
    cache.set("key1", {"result": 1})
    cache.set("key2", {"result": 2})

    # Retrieve results
    assert cache.get("key1") == {"result": 1}
    assert cache.get("key2") == {"result": 2}

    # Test max size enforcement
    cache.set("key3", {"result": 3})
    assert cache.get("key1") is None  # Should be evicted
