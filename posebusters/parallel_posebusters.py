import multiprocessing as mp
import warnings
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union
from pathlib import Path
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor

import pandas as pd
import numpy as np
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

from . import PoseBusters
from .parallel.cache import ComputationCache
from .tools.loading import safe_load_mol, safe_supply_mols
from .posebusters import _dataframe_from_output

class InvalidMoleculeError(Exception):
    """Invalid molecule input error."""
    pass

class InvalidFilePathError(Exception):
    """Invalid file path error."""
    pass

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

    def bust(
        self,
        mol_pred: Union[Iterable[Union[Mol, Path, str]], Mol, Path, str],
        mol_true: Optional[Union[Mol, Path, str]] = None,
        mol_cond: Optional[Union[Mol, Path, str]] = None,
        batch_size: int = 100,
        full_report: bool = False,
        output_intermediate_values: bool = False,
    ) -> pd.DataFrame:
        """Run parallel tests on molecules."""
        # Noneチェック
        if mol_pred is None:
            raise InvalidMoleculeError("mol_pred cannot be None")

        # mol_pred_listの準備と検証
        try:
            mol_pred_list = [mol_pred] if isinstance(mol_pred, (Mol, Path, str)) else list(mol_pred)
        except TypeError as e:
            raise InvalidMoleculeError(f"Invalid input type for mol_pred: {e}")

        # リスト内のNoneチェック
        if any(mol_p is None for mol_p in mol_pred_list):
            raise InvalidMoleculeError("mol_pred list contains None")

        # パスの検証
        for mol_p in mol_pred_list:
            if isinstance(mol_p, (str, Path)) and not Path(mol_p).exists():
                raise InvalidFilePathError(f"File not found: {mol_p}")
        # 参照分子のパス検証
        if isinstance(mol_true, (str, Path)) and not Path(mol_true).exists():
            raise InvalidFilePathError(f"Reference molecule file not found: {mol_true}")
        if isinstance(mol_cond, (str, Path)) and not Path(mol_cond).exists():
            raise InvalidFilePathError(f"Conditioning molecule file not found: {mol_cond}")

        # DataFrameの準備
        columns = ["mol_pred", "mol_true", "mol_cond"]
        self.file_paths = pd.DataFrame(
            [[p, mol_true, mol_cond] for p in mol_pred_list],
            columns=columns
        )

        results_gen = self._run()
        df = pd.concat([_dataframe_from_output(d, self.config, full_report=full_report) for d in results_gen])
        df.index.names = ["file", "molecule"]
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        return df

    def _run(self) -> Generator[dict, None, None]:
        """Run all tests on molecules provided in file paths."""
        self._initialize_modules()

        # 全体の処理数を計算
        total_mols = sum(
            len(list(safe_supply_mols(
                paths["mol_pred"],
                **self.config.get("loading", {}).get("mol_pred", {})
            )))
            for _, paths in self.file_paths.iterrows()
        )

        with tqdm(total=total_mols, disable=not self.show_progress) as pbar:
            for _, paths in self.file_paths.iterrows():
                mol_args = {}
                if "mol_cond" in paths and paths["mol_cond"] is not None:
                    try:
                        mol_cond_load_params = self.config.get("loading", {}).get("mol_cond", {})
                        mol_args["mol_cond"] = safe_load_mol(path=paths["mol_cond"], **mol_cond_load_params)
                        if mol_args["mol_cond"] is None:
                            warnings.warn(f"Failed to load conditioning molecule: {paths['mol_cond']}", UserWarning)
                    except Exception as e:
                        warnings.warn(f"Error loading conditioning molecule: {e}", UserWarning)

                if "mol_true" in paths and paths["mol_true"] is not None:
                    try:
                        mol_true_load_params = self.config.get("loading", {}).get("mol_true", {})
                        mol_args["mol_true"] = safe_load_mol(path=paths["mol_true"], **mol_true_load_params)
                        if mol_args["mol_true"] is None:
                            warnings.warn(f"Failed to load reference molecule: {paths['mol_true']}", UserWarning)
                    except Exception as e:
                        warnings.warn(f"Error loading reference molecule: {e}", UserWarning)

                # 予測分子のバッチ処理
                mol_pred_load_params = self.config.get("loading", {}).get("mol_pred", {})
                try:
                    mols = list(safe_supply_mols(paths["mol_pred"], **mol_pred_load_params))
                except Exception as e:
                    warnings.warn(f"Error loading predicted molecules: {e}", UserWarning)
                    continue

                if self.config["top_n"] is not None:
                    mols = mols[:self.config["top_n"]]

                mol_batches = [
                    mols[i:i + self.batch_size]
                    for i in range(0, len(mols), self.batch_size)
                ]

                for batch in mol_batches:
                    try:
                        # バッチ内の各分子を処理
                        batch_results = self._process_batch(
                            batch=batch,
                            mol_args=mol_args,
                            paths=paths
                        )
                        # 結果を生成
                        for result in batch_results:
                            yield result
                            pbar.update(1)
                    except Exception as e:
                        warnings.warn(f"Error processing batch: {e}", UserWarning)
                        continue

    def _process_batch(
        self,
        batch: List[Mol],
        mol_args: Dict[str, Mol],
        paths: pd.Series
    ) -> List[Dict[Tuple[str, str], List[Tuple[str, str, Any]]]]:
        """Process a batch of molecules in parallel."""
        if not batch:
            return []

        # バッチサイズに応じてワーカー数を調整
        effective_n_workers = min(self.n_workers, len(batch))

        Executor = ThreadPoolExecutor if self.use_threading else ProcessPoolExecutor
        with Executor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(
                    self._process_batch_molecules,
                    batch_chunk,
                    mol_args,
                    paths,
                    start_idx=i * (len(batch) // effective_n_workers)
                )
                for i, batch_chunk in enumerate(
                    np.array_split(batch, self.n_workers)
                )
            ]

            batch_results = []
            for future in as_completed(futures):
                try:
                    results = future.result()
                    if results:
                        batch_results.extend(results)
                except Exception as e:
                    warnings.warn(f"Error in batch {futures[future]}: {str(e)}")

        return batch_results

    def _process_batch_molecules(
        self,
        molecules: List[Mol],
        mol_args: Dict[str, Mol],
        paths: pd.Series,
        start_idx: int
    ) -> List[Dict[Tuple[str, str], List[Tuple[str, str, Any]]]]:
        """Process a chunk of molecules from a batch."""
        results = []
        for i, mol_pred in enumerate(molecules):
            if mol_pred is None:
                continue

            mol_args_copy = mol_args.copy()
            mol_args_copy["mol_pred"] = mol_pred
            results_key = (str(paths["mol_pred"]), self._get_name(mol_pred, start_idx + i))

            result = self._process_single_molecule(mol_args_copy, results_key)
            if result is not None:
                results.append(result)

        return results

    def _process_single_molecule(
        self,
        mol_args: Dict[str, Mol],
        results_key: Tuple[str, str]
    ) -> Dict[Tuple[str, str], List[Tuple[str, str, Any]]]:
        """Process a single molecule with all modules."""
        results = []

        for name, fname, func, args in zip(self.module_name, self.fname, self.module_func, self.module_args):
            cache_key = None
            if "mol_pred" in mol_args:
                cache_key = self.cache.get_key(mol_args["mol_pred"], fname)
                if cache_key:
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        results.extend([(name, k, v) for k, v in cached_result.items()])
                        continue

            args_needed = {k: v for k, v in mol_args.items() if k in args}
            if fname == "loading":
                args_needed = {k: args_needed.get(k, None) for k in args_needed}
            if fname != "loading" and not all(args_needed.get(m, None) for m in args_needed):
                module_output: dict[str, Any] = {"results": {}}
            else:
                module_output = func(**args_needed)
                if cache_key:
                    self.cache.set(cache_key, module_output["results"])

            results.extend([(name, k, v) for k, v in module_output["results"].items()])

        return {results_key: results}
