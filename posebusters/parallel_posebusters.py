import multiprocessing as mp
import warnings
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pandas as pd
import numpy as np
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

from . import PoseBusters
from .parallel.cache import ComputationCache
from .tools.loading import safe_load_mol, safe_supply_mols


class ParallelPoseBusters(PoseBusters):
    """PoseBusters with parallel processing capabilities."""

    def __init__(
        self,
        config: str | dict[str, Any] = "dock",
        n_workers: Optional[int] = None,
        use_threading: bool = False,
        cache_size: int = 1000,
        show_progress: bool = True,
        batch_size: int = 100,
        top_n: int | None = None,
    ):
        """Initialize ParallelPoseBusters."""
        super().__init__(config=config, top_n=top_n)
        self.n_workers = n_workers or mp.cpu_count()
        self.use_threading = use_threading
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.cache = ComputationCache(max_size=cache_size)
        if cache_size < 10:
            warnings.warn(
                "Cache size is very small (< 10). This might impact performance.",
                UserWarning
            )

    def _run(self) -> Generator[dict, None, None]:
        """Run all tests on molecules provided in file paths."""
        self._initialize_modules()
        # Get total mol numbers
        total_mols = sum(
            len(list(safe_supply_mols(
                paths["mol_pred"],
                **self.config.get("loading", {}).get("mol_pred", {})
            )))
            for _, paths in self.file_paths.iterrows()
        )


        with tqdm(total=total_mols, disable=not self.show_progress) as pbar:
            for _, paths in self.file_paths.iterrows():
                # 参照分子の読み込み
                mol_args = {}
                if "mol_cond" in paths and paths["mol_cond"] is not None:
                    mol_cond_load_params = self.config.get("loading", {}).get("mol_cond", {})
                    mol_args["mol_cond"] = safe_load_mol(path=paths["mol_cond"], **mol_cond_load_params)
                if "mol_true" in paths and paths["mol_true"] is not None:
                    mol_true_load_params = self.config.get("loading", {}).get("mol_true", {})
                    mol_args["mol_true"] = safe_load_mol(path=paths["mol_true"], **mol_true_load_params)

                mol_pred_load_params = self.config.get("loading", {}).get("mol_pred", {})
                mols = list(safe_supply_mols(paths["mol_pred"], **mol_pred_load_params))

                if self.config["top_n"] is not None:
                    mols = mols[:self.config["top_n"]]

                mol_batches = [
                    mols[i:i + self.batch_size]
                    for i in range(0, len(mols), self.batch_size)
                ]

                for batch in mol_batches:
                    # バッチ内の各分子を処理
                    batch_results = self._process_batch(
                        batch=batch,
                        mol_args=mol_args,
                        paths=paths
                    )
                    pbar.update(len(batch))

                    # 結果を生成
                    for result in batch_results:
                        yield result

    def _process_batch(
        self,
        batch: List[Mol],
        mol_args: Dict[str, Mol],
        paths: pd.Series
    ) -> List[Dict[Tuple[str, str], List[Tuple[str, str, Any]]]]:
        """Process a batch of molecules."""
        Executor = ThreadPoolExecutor if self.use_threading else ProcessPoolExecutor
        batch_results = []

        with Executor(max_workers=self.n_workers) as executor:
            futures = []
            for i, mol_pred in enumerate(batch):
                if mol_pred is None:
                    continue

                mol_args_copy = mol_args.copy()
                mol_args_copy["mol_pred"] = mol_pred
                results_key = (str(paths["mol_pred"]), self._get_name(mol_pred, i))

                futures.append(
                    executor.submit(
                        self._process_single_molecule,
                        mol_args_copy,
                        results_key
                    )
                )

            for future in futures:
                result = future.result()
                if result is not None:
                    batch_results.append(result)

        return batch_results

    def _process_single_molecule(
        self,
        mol_args: Dict[str, Mol],
        results_key: Tuple[str, str]
    ) -> Dict[Tuple[str, str], List[Tuple[str, str, Any]]]:
        """Process a single molecule with all modules."""
        results = []

        for name, fname, func, args in zip(self.module_name, self.fname, self.module_func, self.module_args):
            # キャッシュのチェック
            cache_key = None
            if "mol_pred" in mol_args:
                cache_key = self.cache.get_key(mol_args["mol_pred"], fname)
                if cache_key:
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        results.extend([(name, k, v) for k, v in cached_result.items()])
                        continue

            # 必要な引数の準備
            args_needed = {k: v for k, v in mol_args.items() if k in args}
            if fname == "loading":
                args_needed = {k: args_needed.get(k, None) for k in args_needed}

            # モジュールの実行
            if fname != "loading" and not all(args_needed.get(m, None) for m in args_needed):
                module_output: dict[str, Any] = {"results": {}}
            else:
                module_output = func(**args_needed)
                if cache_key:
                    self.cache.set(cache_key, module_output["results"])

            # 結果の保存
            results.extend([(name, k, v) for k, v in module_output["results"].items()])

        return {results_key: results}
