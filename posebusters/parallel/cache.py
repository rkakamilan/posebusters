from typing import Any, Optional
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

class ComputationCache:
    """Cache for storing computation results of molecules."""
    def __init__(self, max_size: int = 1000):
        """Initialize cache with maximum size.
        Args:
            max_size: Maximum number of results to cache
        """
        self.cache = {}
        self.max_size = max_size

    def get_key(self, mol: Mol, computation: str) -> Optional[str]:
        """Generate cache key from molecule and computation type."""
        try:
            inchi = Chem.MolToInchi(mol)
            return f"{inchi}_{computation}"
        except:
            return None

    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store result in cache."""
        if len(self.cache) >= self.max_size:
            while len(self.cache) >= self.max_size:
                self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
