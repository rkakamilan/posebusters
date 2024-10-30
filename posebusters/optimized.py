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

    # def _parallel_process_molecule(
    #     self,
    #     mol_args: Dict[str, Mol],
    #     module_configs: List[Tuple]
    # ) -> Dict:
    #     """Process single molecule with all modules."""
    #     results = {}

    #     # Early return for invalid molecules
    #     if not all(mol is not None for mol in mol_args.values()):
    #         return {"results": {}}

    #     for name, fname, func, args in module_configs:
    #         # Check cache first
    #         cache_key = self.cache.get_key(mol_args["mol_pred"], fname)
    #         if cache_key:
    #             cached_result = self.cache.get(cache_key)
    #             if cached_result is not None:
    #                 results[name] = cached_result
    #                 continue

    #         # Run computation
    #         args_needed = {k: v for k, v in mol_args.items() if k in args}
    #         try:
    #             if fname == "loading":
    #                 args_needed = {k: args_needed.get(k, None) for k in args_needed}
    #             module_output = func(**args_needed)
    #             results[name] = module_output["results"]

    #             # Cache result
    #             if cache_key:
    #                 self.cache.set(cache_key, module_output["results"])
    #         except Exception as e:
    #             warnings.warn(f"Error in {name}: {str(e)}")
    #             results[name] = {}

    #     return results

    def _parallel_process_molecule(
        self,
        mol_args: Dict[str, Mol],
        module_configs: List[tuple]
    ) -> Dict[str, Any]:
        """Process single molecule with all modules and flatten results."""
        # 早期リターン
        if not all(mol_args.get(k) is not None for k in mol_args):
            return self._get_empty_results()

        flattened_results = {}
        
        # 各モジュールを実行
        for name, fname, func, args in module_configs:
            # キャッシュをチェック
            cache_key = None
            if "mol_pred" in mol_args:
                cache_key = self.cache.get_key(mol_args["mol_pred"], fname)
                if cache_key:
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        flattened_results.update(cached_result)
                        continue

            # 計算実行
            args_needed = {k: v for k, v in mol_args.items() if k in args}
            try:
                if fname == "loading":
                    args_needed = {k: args_needed.get(k, None) for k in args_needed}
                module_output = func(**args_needed)
                
                # 結果をフラット化して追加
                if "results" in module_output:
                    flattened_results.update(module_output["results"])
                    
                    # 結果をキャッシュ
                    if cache_key:
                        self.cache.set(cache_key, module_output["results"])
                        
            except Exception as e:
                warnings.warn(f"Error in {name}: {str(e)}")
                # エラー時は該当モジュールの出力をNaNで埋める
                for output in self._get_module_outputs(fname):
                    flattened_results[output] = np.nan
                    
        return flattened_results

    def _get_empty_results(self) -> Dict[str, Any]:
        """Get empty results with all expected columns."""
        empty_results = {}
        for module in self.config["modules"]:
            for output in module.get("chosen_binary_test_output", []):
                empty_results[output] = np.nan
        return empty_results

    def _get_module_outputs(self, fname: str) -> List[str]:
        """Get expected outputs for a given module."""
        for module in self.config["modules"]:
            if module["function"] == fname:
                return module.get("chosen_binary_test_output", [])
        return []

    def bust(
        self,
        mol_pred: Union[Iterable[Union[Mol, Path, str]], Mol, Path, str],
        mol_true: Optional[Union[Mol, Path, str]] = None,
        mol_cond: Optional[Union[Mol, Path, str]] = None,
        batch_size: int = 100,
        full_report: bool = False,
    ) -> pd.DataFrame:
        """Run optimized tests on molecules."""
        # 分子リストの準備
        mol_pred_list = [mol_pred] if isinstance(mol_pred, (Mol, Path, str)) else list(mol_pred)
        
        # ファイルパスの設定
        columns = ["mol_pred", "mol_true", "mol_cond"]
        self.file_paths = pd.DataFrame(
            [[p, mol_true, mol_cond] for p in mol_pred_list],
            columns=columns
        )

        # モジュール設定の準備
        module_configs = list(zip(
            self.module_name,
            self.fname,
            self.module_func,
            self.module_args
        ))

        # バッチ処理
        results = []
        mol_batches = [
            mol_pred_list[i:i + batch_size] 
            for i in range(0, len(mol_pred_list), batch_size)
        ]
        
        with tqdm(total=len(mol_pred_list), disable=not self.show_progress) as pbar:
            for batch in mol_batches:
                # バッチの分子引数を準備
                mol_args_batch = []
                for mol_p in batch:
                    args = {"mol_pred": self._load_molecule(mol_p)}
                    if mol_true is not None:
                        args["mol_true"] = self._load_molecule(mol_true)
                    if mol_cond is not None:
                        args["mol_cond"] = self._load_molecule(mol_cond)
                    mol_args_batch.append(args)
                
                # バッチ処理
                batch_results = []
                for mol_args in mol_args_batch:
                    result = self._parallel_process_molecule(mol_args, module_configs)
                    batch_results.append(result)
                results.extend(batch_results)
                pbar.update(len(batch))

        # 結果をDataFrameに変換
        df = pd.DataFrame(results)
        
        if len(df) == 0:
            # 空の結果の場合、期待される列を持つ空のDataFrameを返す
            empty_results = self._get_empty_results()
            df = pd.DataFrame([empty_results])

        # インデックスの設定
        df.index = pd.MultiIndex.from_product(
            [[str(p) for p in mol_pred_list], [""]],
            names=["file", "molecule"]
        )

        if not full_report:
            # 名前変更マッピングの作成
            rename_map = {}
            for module in self.config["modules"]:
                rename_map.update(module.get("rename_outputs", {}))
            
            # 列名の変更を適用
            df = df.rename(columns=rename_map)

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
