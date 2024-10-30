from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from functools import lru_cache, partial
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .core.cache import ComputationCache
from .core.parallel import process_batch
from .core.optimization import ModuleRunner
from . import PoseBusters
from .tools.loading import safe_load_mol


class OptimizedPoseBusters(PoseBusters):
    """Optimized version of PoseBusters with parallel processing and caching."""
    
    def __init__(
        self, 
        config: Union[str, Dict[str, Any]] = "redock",
        n_workers: Optional[int] = None,
        use_threading: bool = False,
        cache_size: int = 1000,
        show_progress: bool = True
    ):
        """Initialize OptimizedPoseBusters.
        
        Args:
            config: Configuration string or dictionary
            n_workers: Number of worker processes/threads
            use_threading: Use threads instead of processes
            cache_size: Size of computation cache
            show_progress: Show progress bar
        """
        super().__init__(config=config)
        self.n_workers = n_workers or mp.cpu_count()
        self.use_threading = use_threading
        self.cache = ComputationCache(max_size=cache_size)
        self.show_progress = show_progress
        self.module_runner = ModuleRunner()
        self._initialize_modules()

    def _parallel_process_molecule(
        self,
        mol_args: Dict[str, Mol],
        module_configs: List[Tuple]
    ) -> Dict:
        """Process single molecule with all modules."""
        results = {}

        # Early return for invalid molecules
        if not all(mol is not None for mol in mol_args.values()):
            return {"results": {}}

        for name, fname, func, args in module_configs:
            # Check cache first
            cache_key = self.cache.get_key(mol_args["mol_pred"], fname)
            if cache_key:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    results[name] = cached_result
                    continue

            # Run computation
            args_needed = {k: v for k, v in mol_args.items() if k in args}
            try:
                if fname == "loading":
                    args_needed = {k: args_needed.get(k, None) for k in args_needed}
                module_output = func(**args_needed)
                results[name] = module_output["results"]

                # Cache result
                if cache_key:
                    self.cache.set(cache_key, module_output["results"])
            except Exception as e:
                warnings.warn(f"Error in {name}: {str(e)}")
                results[name] = {}

        return results

    def _process_batch(
        self,
        mol_batch: List[Dict[str, Mol]],
        module_configs: List[Tuple]
    ) -> List[Dict]:
        """Process a batch of molecules in parallel."""
        process_func = partial(self._parallel_process_molecule, module_configs=module_configs)
        
        Executor = ThreadPoolExecutor if self.use_threading else ProcessPoolExecutor
        with Executor(max_workers=self.n_workers) as executor:
            results = list(executor.map(process_func, mol_batch))
        return results

    def bust(
        self,
        mol_pred: Union[Iterable[Union[Mol, Path, str]], Mol, Path, str],
        mol_true: Optional[Union[Mol, Path, str]] = None,
        mol_cond: Optional[Union[Mol, Path, str]] = None,
        batch_size: int = 100,
        full_report: bool = False,
    ) -> pd.DataFrame:
        """Run optimized tests on molecules.
        
        Args:
            mol_pred: Predicted molecule(s)
            mol_true: True molecule
            mol_cond: Conditioning molecule
            batch_size: Size of parallel processing batches
            full_report: Include full report
            
        Returns:
            DataFrame with results
        """
        # Prepare molecule list
        mol_pred_list = [mol_pred] if isinstance(mol_pred, (Mol, Path, str)) else mol_pred
        
        # Prepare module configurations
        module_configs = list(zip(
            self.module_name,
            self.fname,
            self.module_func,
            self.module_args
        ))
        
        # Process molecules in batches
        results = []
        mol_batches = [mol_pred_list[i:i + batch_size] 
                      for i in range(0, len(mol_pred_list), batch_size)]
        
        with tqdm(total=len(mol_pred_list), disable=not self.show_progress) as pbar:
            for batch in mol_batches:
                # Prepare molecule arguments for batch
                mol_args_batch = []
                for mol_p in batch:
                    args = {"mol_pred": self._load_molecule(mol_p)}
                    if mol_true is not None:
                        args["mol_true"] = self._load_molecule(mol_true)
                    if mol_cond is not None:
                        args["mol_cond"] = self._load_molecule(mol_cond)
                    mol_args_batch.append(args)
                
                # Process batch
                batch_results = self._process_batch(mol_args_batch, module_configs)
                results.extend(batch_results)
                pbar.update(len(batch))
        

        df = pd.DataFrame(results)
        df.index.names = ["file", "molecule"]
        if not full_report:
            # 設定ファイルから選択された出力列を取得
            chosen_columns = []
            rename_map = {}
            for module in self.config["modules"]:
                # 選択された出力列を収集
                binary_outputs = module.get("chosen_binary_test_output", [])
                chosen_columns.extend(binary_outputs)
                # 出力列の名前変更マッピングを収集
                rename_outputs = module.get("rename_outputs", {})
                rename_map.update(rename_outputs)

            # 選択された列のみを保持し、名前を変更
            df = df[chosen_columns]
            df.columns = [rename_map.get(col, col) for col in df.columns]
        # if not full_report:
        #     df = df[self.config["output_columns"]]
        
        return df

    @staticmethod
    @lru_cache(maxsize=1000)
    def _load_molecule(mol_input: Union[Mol, Path, str]) -> Optional[Mol]:
        """Load molecule with caching."""
        if isinstance(mol_input, Mol):
            return mol_input
        try:
            return safe_load_mol(mol_input)
        except Exception as e:
            warnings.warn(f"Failed to load molecule: {str(e)}")
            return None
