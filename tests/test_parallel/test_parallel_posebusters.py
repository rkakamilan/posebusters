import pytest
import pandas as pd
import numpy as np

from rdkit.Chem import MolFromMolFile
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from posebusters import ParallelPoseBusters
from posebusters.parallel_posebusters import InvalidMoleculeError, InvalidFilePathError


@pytest.fixture
def parallel_pb():
    """Basic ParallelPoseBusters instance."""
    return ParallelPoseBusters(
        config="mol",
        n_workers=2,
        batch_size=2,
        show_progress=False
    )

@pytest.fixture
def simple_mol(shared_datadir):
    """Simple molecule fixture."""
    return MolFromMolFile(str(shared_datadir / "mol_PM2.sdf"))

@pytest.fixture
def mol_list(shared_datadir):
    """List of molecules for batch testing."""
    mol_files = [
        "mol_PM2.sdf",
        "mol_CGB.sdf",
        "mol_RQ3_x00.sdf",
    ]
    return [MolFromMolFile(str(shared_datadir / f)) for f in mol_files]

def test_initialization():
    """Test initialization with different parameters."""
    pb = ParallelPoseBusters()
    assert isinstance(pb.n_workers, int)  # mp.cpu_count() is set as default
    assert not pb.use_threading
    assert pb.batch_size == 100
    assert pb.show_progress

    pb = ParallelPoseBusters(
        config="mol",
        n_workers=4,
        use_threading=True,
        batch_size=50,
        show_progress=False
    )
    assert pb.n_workers == 4
    assert pb.use_threading
    assert pb.batch_size == 50
    assert not pb.show_progress

def test_cache_warning():
    """Test warning for small cache size."""
    with pytest.warns(UserWarning) as record:
        ParallelPoseBusters(cache_size=5)
        # warnings.warn("Cache size is very small")
        assert len(record) == 1

def test_single_molecule_processing(parallel_pb, simple_mol):
    """Test processing of a single molecule."""
    result = parallel_pb.bust(simple_mol)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.index.names == ["file", "molecule"]

def test_batch_processing(parallel_pb, mol_list):
    """Test batch processing of multiple molecules."""
    # バッチサイズ2で3つの分子を処理
    result = parallel_pb.bust(mol_list)
    assert len(result) == len(mol_list)
    assert not result.isna().all().all()

def test_threading_vs_multiprocessing(simple_mol):
    """Test both threading and multiprocessing execution."""
    # スレッド使用
    pb_thread = ParallelPoseBusters(
        config="mol",
        n_workers=2,
        use_threading=True,
        show_progress=False
    )
    result_thread = pb_thread.bust(simple_mol)

    # プロセス使用
    pb_process = ParallelPoseBusters(
        config="mol",
        n_workers=2,
        use_threading=False,
        show_progress=False
    )
    # molecule columns is not mol's '_Name' prop
    result_process = pb_process.bust(simple_mol)

    # 結果が同じになることを確認
    # shared_datadir のtempdirが毎回異なるためfileは違うpathになる
    assert result_thread.columns.equals(result_process.columns)
    assert result_thread.shape == result_process.shape
    pd.testing.assert_frame_equal(
        result_thread.reset_index().drop(["file", "molecule"], axis=1),
        result_process.reset_index().drop(["file", "molecule"], axis=1),
    )

def test_cache_effectiveness(parallel_pb, simple_mol):
    """Test that caching improves performance."""
    # キャッシュの状態を確認するためのヘルパー関数
    def get_cache_size(pb):
        return len(pb.cache.cache)

    # 1回目の実行
    result1 = parallel_pb.bust(simple_mol)
    cache_size1 = get_cache_size(parallel_pb)

    # 2回目の実行
    result2 = parallel_pb.bust(simple_mol)
    cache_size2 = get_cache_size(parallel_pb)

    assert cache_size2 >= cache_size1
    # pd.testing.assert_frame_equal(result1, result2)
    # shared_datadir のtempdirが毎回異なるためfileは違うpathになる
    pd.testing.assert_frame_equal(
        result1.reset_index().drop(["file"], axis=1),
        result2.reset_index().drop(["file"], axis=1),
    )

def test_error_handling(parallel_pb, simple_mol):
    """Test error handling for invalid inputs."""
    # Noneの処理
    with pytest.raises(InvalidMoleculeError, match="mol_pred cannot be None"):
        parallel_pb.bust(None)

    # 存在しないファイルの処理
    with pytest.raises(InvalidFilePathError, match="File not found"):
        parallel_pb.bust("non_existent.sdf")

    # 無効な入力タイプの処理
    with pytest.raises(InvalidMoleculeError, match="Invalid input type"):
        parallel_pb.bust(123)

    # 無効な参照分子の処理
    with pytest.raises(InvalidFilePathError, match="Reference molecule file not found"):
        parallel_pb.bust(
            mol_pred=simple_mol,
            mol_true="invalid_ref.sdf",
        )

def test_mixed_input_handling(parallel_pb, simple_mol):
    """Test that invalid molecules in a list cause early failure."""
    # 有効な分子と無効なパスの混合
    inputs = [
        simple_mol,
        None,
        "non_existent.sdf",
        simple_mol,
    ]
    # Noneを含むリストは早期に失敗
    with pytest.raises(InvalidMoleculeError):
        parallel_pb.bust(inputs)

def test_batch_size_effects(parallel_pb, mol_list):
    """Test different batch sizes."""
    batch_sizes = [1, 2, len(mol_list)]
    results = []

    for batch_size in batch_sizes:
        parallel_pb.batch_size = batch_size
        result = parallel_pb.bust(mol_list)
        results.append(result)

    for i in range(1, len(results)):
        pd.testing.assert_frame_equal(
            results[0].reset_index().drop(["file"], axis=1),
            results[i].reset_index().drop(["file"], axis=1),
        )

def test_parallel_processing_with_references(parallel_pb, shared_datadir):
    """Test parallel processing with reference molecules."""
    mol_pred = shared_datadir / "mol_PM2.sdf"
    mol_true = shared_datadir / "mol_PM2.sdf"  # 同じ分子を参照として使用
    mol_cond = shared_datadir / "mol_PM2.sdf"  # 同じ分子を条件として使用

    result = parallel_pb.bust(
        mol_pred=mol_pred,
        mol_true=mol_true,
        mol_cond=mol_cond
    )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.index.names == ["file", "molecule"]

def test_compatibility_with_original_output(parallel_pb, simple_mol):
    """Test compatibility with original PoseBusters output."""
    from posebusters import PoseBusters

    original_pb = PoseBusters(config="mol")
    original_result = original_pb.bust(simple_mol)

    parallel_result = parallel_pb.bust(simple_mol)

    assert set(original_result.columns) == set(parallel_result.columns)
    assert original_result.dtypes.equals(parallel_result.dtypes)

def test_top_n_limitation(parallel_pb, mol_list, mol_pred_1jn2_gen3):
    """Test top_n limitation functionality."""
    top_n = 1
    pb_with_limit = ParallelPoseBusters(
        config="mol",
        n_workers=2,
        show_progress=False,
        top_n=top_n
    )

    result = pb_with_limit.bust(mol_list)
    assert len(result) == len(mol_list)

    result = pb_with_limit.bust(mol_pred_1jn2_gen3)
    assert len(result) == top_n
