from typing import Any, Dict, List, Optional
import numpy as np
from rdkit.Chem.rdchem import Mol

class MoleculeProcessor:
    """Processes molecules with caching support."""

    def process_molecule(
        self,
        mol_args: Dict[str, Mol],
        module_configs: List[tuple],
        cache: Optional['ComputationCache'] = None
    ) -> Dict[str, Any]:
        """Process single molecule with all modules."""
        # Early return for invalid molecules
        if not all(mol_args.get(k) is not None for k in mol_args):
            return self.get_empty_results(module_configs)

        results = {}

        for name, fname, func, args in module_configs:
            # Check cache
            cache_key = None
            if cache and "mol_pred" in mol_args:
                cache_key = cache.get_key(mol_args["mol_pred"], fname)
                if cache_key:
                    cached_result = cache.get(cache_key)
                    if cached_result is not None:
                        results.update(cached_result)
                        continue

            # Run computation
            try:
                args_needed = {k: v for k, v in mol_args.items() if k in args}
                if fname == "loading":
                    args_needed = {k: args_needed.get(k, None) for k in args_needed}
                    
                module_output = func(**args_needed)
                if "results" in module_output:
                    results.update(module_output["results"])
                    
                    # Cache result
                    if cache_key:
                        cache.set(cache_key, module_output["results"])
                        
            except Exception as e:
                warnings.warn(f"Error in {name}: {str(e)}")
                # Fill failed module outputs with NaN
                for output in self.get_module_outputs(module_configs, fname):
                    results[output] = np.nan
                    
        return results

    @staticmethod
    def get_empty_results(module_configs: List[tuple]) -> Dict[str, Any]:
        """Get empty results dictionary."""
        empty_results = {}
        for _, fname, _, _ in module_configs:
            for output in MoleculeProcessor.get_module_outputs(module_configs, fname):
                empty_results[output] = np.nan
        return empty_results

    @staticmethod
    def get_module_outputs(module_configs: List[tuple], fname: str) -> List[str]:
        """Get expected outputs for a module."""
        for _, module_fname, _, outputs in module_configs:
            if module_fname == fname:
                return outputs
        return []