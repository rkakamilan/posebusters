from typing import Any, Dict, Iterable, List, Optional, Union
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

from . import PoseBusters
from .parallel.cache import ComputationCache
from .parallel.executor import ParallelExecutor
from .parallel.processor import MoleculeProcessor
from .tools.loading import safe_load_mol

class ParallelPoseBusters(PoseBusters):
    """PoseBusters with parallel processing capabilities."""
    
    def __init__(
        self, 
        config: Union[str, Dict[str, Any]] = "redock",
        n_workers: Optional[int] = None,
        use_threading: bool = False,
        cache_size: int = 1000,
        show_progress: bool = True
    ):
        """Initialize ParallelPoseBusters.
        
        Args:
            config: Configuration string or dictionary
            n_workers: Number of worker processes/threads
            use_threading: Use threads instead of processes
            cache_size: Size of computation cache
            show_progress: Show progress bar
        """
        super().__init__(config=config)
        
        self.executor = ParallelExecutor(
            n_workers or mp.cpu_count(),
            use_threading=use_threading
        )
        self.cache = ComputationCache(max_size=cache_size)
        self.processor = MoleculeProcessor()
        self.show_progress = show_progress
        
        self._initialize_modules()

    def bust(
        self,
        mol_pred: Union[Iterable[Union[Mol, Path, str]], Mol, Path, str],
        mol_true: Optional[Union[Mol, Path, str]] = None,
        mol_cond: Optional[Union[Mol, Path, str]] = None,
        batch_size: int = 100,
        full_report: bool = False,
    ) -> pd.DataFrame:
        """Run parallel tests on molecules."""
        # Prepare molecule list
        mol_pred_list = [mol_pred] if isinstance(mol_pred, (Mol, Path, str)) else list(mol_pred)
        
        # Prepare file paths
        columns = ["mol_pred", "mol_true", "mol_cond"]
        self.file_paths = pd.DataFrame(
            [[p, mol_true, mol_cond] for p in mol_pred_list],
            columns=columns
        )

        # Prepare module configurations
        module_configs = list(zip(
            self.module_name,
            self.fname,
            self.module_func,
            self.module_args
        ))

        # Process in batches
        results = []
        mol_batches = [
            mol_pred_list[i:i + batch_size] 
            for i in range(0, len(mol_pred_list), batch_size)
        ]
        
        with tqdm(total=len(mol_pred_list), disable=not self.show_progress) as pbar:
            for batch in mol_batches:
                # Prepare batch arguments
                mol_args_batch = []
                for mol_p in batch:
                    args = {"mol_pred": safe_load_mol(mol_p)}
                    if mol_true is not None:
                        args["mol_true"] = safe_load_mol(mol_true)
                    if mol_cond is not None:
                        args["mol_cond"] = safe_load_mol(mol_cond)
                    mol_args_batch.append(args)

                # Process batch
                batch_results = self.executor.execute_batch(
                    mol_args_batch,
                    self.processor.process_molecule,
                    module_configs=module_configs,
                    cache=self.cache
                )
                results.extend(batch_results)
                pbar.update(len(batch))

        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        if len(df) == 0:
            # Return empty DataFrame with expected columns
            empty_results = self.processor.get_empty_results(module_configs)
            df = pd.DataFrame([empty_results])

        # Set index
        df.index = pd.MultiIndex.from_product(
            [[str(p) for p in mol_pred_list], [""]],
            names=["file", "molecule"]
        )

        if not full_report:
            # Apply column renaming
            rename_map = {}
            for module in self.config["modules"]:
                rename_map.update(module.get("rename_outputs", {}))
            df = df.rename(columns=rename_map)

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        return df
