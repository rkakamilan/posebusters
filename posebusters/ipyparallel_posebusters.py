import warnings
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import ipyparallel as ipp
import pandas as pd
import numpy as np

from rdkit.Chem.rdchem import Mol
from tqdm.auto import tqdm

from . import PoseBusters
from .parallel.cache import ComputationCache
from .tools.loading import safe_load_mol, safe_supply_mols
from .posebusters import _dataframe_from_output

class IPyParallelPoseBusters(PoseBusters):
    """PoseBusters with IPyParallel processing capabilities."""

    def __init__(
        self,
        config: str | dict[str, Any] = "dock",
        profile: str = 'default',
        n_engines: Optional[int] = None,
        cache_size: int = 1000,
        show_progress: bool = True,
        batch_size: int = 100,
        top_n: int | None = None,
    ):
        """Initialize IPyParallelPoseBusters.

        Args:
            config: Configuration for PoseBusters
            profile: IPython profile to use
            n_engines: Number of engines to request
            cache_size: Size of computation cache
            show_progress: Whether to show progress bar
            batch_size: Size of batches for processing
            top_n: Number of top poses to analyze
        """
        super().__init__(config=config, top_n=top_n)
        self.profile = profile
        self.n_engines = n_engines
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.cache = ComputationCache(max_size=cache_size)

        # Initialize cluster connection
        self.cluster = None
        self.direct_view = None
        self.load_balanced_view = None

#     def __enter__(self):
#         """Set up IPyParallel cluster."""
#         try:
#             self.cluster = ipp.Client(profile=self.profile)
#             if self.n_engines is not None:
#                 self.cluster.wait_for_engines(self.n_engines)

#             # Initialize views
#             self.direct_view = self.cluster[:]
#             self.load_balanced_view = self.cluster.load_balanced_view()

#             # Import necessary modules on all engines
#             self.direct_view.execute('import rdkit')
#             self.direct_view.execute('import numpy as np')

#             # Push configuration to all engines
#             self.direct_view.push({'config': self.config})

#             return self

#         except Exception as e:
#             warnings.warn(f"Failed to initialize IPyParallel cluster: {str(e)}")
#             self.cleanup()
#             raise
    def __enter__(self):
        """Set up IPyParallel cluster."""
        try:
            self.cluster = ipp.Client(profile=self.profile)
            if self.n_engines is not None:
                self.cluster.wait_for_engines(self.n_engines)

            # エンジンビューの初期化
            self.direct_view = self.cluster[:]
            self.load_balanced_view = self.cluster.load_balanced_view()

            # 必要なモジュールとクラスをエンジンにプッシュ
            # 変更: モジュールとクラスの依存関係をエンジンに配布
            self.direct_view.execute('import rdkit')
            self.direct_view.execute('import numpy as np')
            self.direct_view.execute('from rdkit import Chem')

            # 変更: PoseBusters関連の設定と関数をエンジンに配布
            self.direct_view.push({
                'config': self.config,
                # 'module_func': self.module_func,
                # 'molecule_args': self.molecule_args
            })

            return self
        except Exception as e:
            warnings.warn(f"Failed to initialize IPyParallel cluster: {str(e)}")
            self.cleanup()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up IPyParallel cluster."""
        self.cleanup()

    def cleanup(self):
        """Clean up cluster resources."""
        if self.cluster is not None:
            self.cluster.close()
            self.cluster = None
            self.direct_view = None
            self.load_balanced_view = None

#     def _process_single_molecule(
#         self,
#         mol_args: Dict[str, Mol],
#         results_key: Tuple[str, str]
#     ) -> Dict[Tuple[str, str], List[Tuple[str, str, Any]]]:
#         """Process a single molecule with all modules."""
#         results = []
#         for name, fname, func, args in zip(
#             self.module_name, self.fname, self.module_func, self.module_args
#         ):
#             cache_key = None
#             if "mol_pred" in mol_args:
#                 cache_key = self.cache.get_key(mol_args["mol_pred"], fname)
#                 if cache_key:
#                     cached_result = self.cache.get(cache_key)
#                     if cached_result is not None:
#                         results.extend([(name, k, v) for k, v in cached_result.items()])
#                         continue
#             args_needed = {k: v for k, v in mol_args.items() if k in args}
#             if fname == "loading":
#                 args_needed = {k: args_needed.get(k, None) for k in args_needed}
#             if fname != "loading" and not all(args_needed.get(m, None) for m in args_needed):
#                 module_output: dict[str, Any] = {"results": {}}
#             else:
#                 module_output = func(**args_needed)
#                 if cache_key:
#                     self.cache.set(cache_key, module_output["results"])
#             results.extend([(name, k, v) for k, v in module_output["results"].items()])

#         return {results_key: results}

#     def _process_batch(
#         self,
#         batch: List[Mol],
#         mol_args: Dict[str, Mol],
#         paths: pd.Series
#     ) -> List[Dict[Tuple[str, str], List[Tuple[str, str, Any]]]]:
#         """Process a batch of molecules in parallel using IPyParallel."""
#         if not batch or self.load_balanced_view is None:
#             return []
#         # Create tasks for each molecule in batch
#         tasks = []
#         for i, mol_pred in enumerate(batch):
#             if mol_pred is None:
#                 continue

#             mol_args_copy = mol_args.copy()
#             mol_args_copy["mol_pred"] = mol_pred
#             results_key = (str(paths["mol_pred"]), self._get_name(mol_pred, i))

#             tasks.append((mol_args_copy, results_key))

#         # Submit tasks to cluster
#         async_results = []
#         for task in tasks:
#             ar = self.load_balanced_view.apply_async(self._process_single_molecule, *task)
#             async_results.append(ar)

#         # Collect results
#         results = []
#         for ar in async_results:
#             try:
#                 result = ar.get()
#                 if result is not None:
#                     results.append(result)
#             except Exception as e:
#                 warnings.warn(f"Error in parallel execution: {str(e)}")

#         return results

    def _run(self) -> Generator[dict, None, None]:
        """Run all tests on molecules provided in file paths."""
        # 変更: 分散処理用の初期化
        self._initialize_modules()
        
        # 進捗表示の設定
        total_mols = sum(1 for _ in self.file_paths.iterrows())
        pbar = tqdm(total=total_mols, disable=not self.show_progress)

        # 変更: バッチ処理の実装
        current_batch = []
        current_paths = []
        
        for _, paths in self.file_paths.iterrows():
            mol_args = self._prepare_molecule_args(paths)
            current_batch.append(mol_args)
            current_paths.append(paths)
            
            # バッチサイズに達したら処理を実行
            if len(current_batch) >= self.batch_size:
                yield from self._process_parallel_batch(current_batch, current_paths)
                current_batch = []
                current_paths = []
                pbar.update(self.batch_size)
        
        # 残りのバッチを処理
        if current_batch:
            yield from self._process_parallel_batch(current_batch, current_paths)
            pbar.update(len(current_batch))
        
        pbar.close()

    def _process_parallel_batch(
        self,
        batch: List[Dict[str, Mol]],
        paths: List[pd.Series]
    ) -> Generator[dict, None, None]:
        """Process a batch of molecules in parallel."""
        # 変更: バッチ処理を分散実行
        if not self.load_balanced_view:
            raise RuntimeError("IPyParallel cluster not initialized")

        # バッチタスクの作成
        tasks = []
        for mol_args, path in zip(batch, paths):
            task = {
                'mol_args': mol_args,
                'module_name': self.module_name,
                'fname': self.fname,
                'func': self.module_func,
                'module_args': self.module_args,
                'path': path,
            }
            tasks.append(task)

        # 変更: 分散処理の実行
        async_results = self.load_balanced_view.map_async(self._remote_process_molecule, tasks)
        
        try:
            results = async_results.get()
            for result in results:
                if result is not None:
                    yield result
        except Exception as e:
            warnings.warn(f"Error in parallel execution: {str(e)}")

    @staticmethod
    def _remote_process_molecule(task: Dict[str, Any]) -> Optional[dict]:
        """リモートエンジンで実行される分子処理関数"""
        try:
            mol_args = task['mol_args']
            module_name = task['module_name']
            fnames = task['fname']
            funcs = task['func']
            module_args = task['module_args']
            path = task['path']

            results = []
            mol_pred = mol_args.get('mol_pred')
            if mol_pred is None:
                return None

            results_key = (str(path['mol_pred']), mol_pred.GetProp('_Name') if mol_pred.HasProp('_Name') else 'unnamed')

            # モジュール実行
            for name, fname, func, args in zip(module_name, fnames, funcs, module_args):
                args_needed = {k: v for k, v in mol_args.items() if k in args}
                if fname == 'loading':
                    args_needed = {k: args_needed.get(k, None) for k in args_needed}
                
                if fname != 'loading' and not all(args_needed.get(m, None) for m in args_needed):
                    module_output = {"results": {}}
                else:
                    module_output = func(**args_needed)

                results.extend([(name, k, v) for k, v in module_output["results"].items()])

            return {results_key: results}

        except Exception as e:
            warnings.warn(f"Error processing molecule: {str(e)}")
            return None

    def _prepare_molecule_args(self, paths: pd.Series) -> Dict[str, Any]:
        """分子引数の準備"""
        # 変更: 分子データの準備処理を分離
        mol_args = {}
        for key in ['mol_cond', 'mol_true', 'mol_pred']:
            if key in paths and paths[key] is not None:
                mol_args[key] = safe_load_mol(
                    path=paths[key],
                    **self.config.get("loading", {}).get(key, {})
                )
        return mol_args

    def bust(
        self,
        mol_pred: Union[Iterable[Union[Mol, Path, str]], Mol, Path, str],
        mol_true: Optional[Union[Mol, Path, str]] = None,
        mol_cond: Optional[Union[Mol, Path, str]] = None,
        batch_size: Optional[int] = None,
        full_report: bool = False,
    ) -> pd.DataFrame:
        """Run parallel tests on molecules."""
        if mol_pred is None:
            raise ValueError("mol_pred cannot be None")

        batch_size = batch_size or self.batch_size

        try:
            with self:  # Setup IPyParallel cluster
                # Prepare molecule list
                mol_pred_list = (
                    [mol_pred] if isinstance(mol_pred, (Mol, Path, str))
                    else list(mol_pred)
                )

                # Create DataFrame with file paths
                columns = ["mol_pred", "mol_true", "mol_cond"]
                self.file_paths = pd.DataFrame(
                    [[p, mol_true, mol_cond] for p in mol_pred_list],
                    columns=columns
                )

                results_gen = self._run()
                df = pd.concat([
                    _dataframe_from_output(d, self.config, full_report=full_report) 
                    for d in results_gen
                ])

                df.index.names = ["file", "molecule"]
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                return df

        except Exception as e:
            warnings.warn(f"Error during parallel execution: {str(e)}")
            raise
