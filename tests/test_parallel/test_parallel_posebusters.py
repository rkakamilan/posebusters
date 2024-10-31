import pytest
from pathlib import Path


# import numpy as np
import pandas as pd
from rdkit.Chem import MolFromMolFile
from posebusters import ParallelPoseBusters


@pytest.fixture
def simple_mol(shared_datadir):
    """Load a simple molecule for testing."""
    return MolFromMolFile(str(shared_datadir / "mol_PM2.sdf"))


@pytest.fixture
def mol_list(shared_datadir):
    """Load a list of test molecules."""
    mol_files = [
        "mol_PM2.sdf",
        "mol_CGB.sdf",
        "mol_RQ3_x00.sdf",
    ]
    return [MolFromMolFile(str(shared_datadir / f)) for f in mol_files]


def test_parallel_posebusters_initialization():
    """Test ParallelPoseBusters initialization with different configurations."""
    # Default initialization
    pb = ParallelPoseBusters()
    assert pb is not None
    assert hasattr(pb, 'n_workers')
    assert hasattr(pb, 'use_threading')

    # Custom initialization
    pb = ParallelPoseBusters(
        config="redock",
        n_workers=2,
        use_threading=True,
        cache_size=500
    )
    assert pb.n_workers == 2
    assert pb.use_threading is True


def test_single_molecule_processing(simple_mol):
    """Test processing a single molecule."""
    pb = ParallelPoseBusters(config="mol")
    result = pb.bust(simple_mol)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "mol_pred_loaded" in result.columns


def test_serial_molecule_processing(mol_list):
    """Test processing molecules one by one."""
    pb = ParallelPoseBusters(config="mol")

    results = []
    for mol in mol_list:
        result = pb.bust(mol)
        results.append(result)

    # Check results
    assert len(results) == len(mol_list)
    assert all(isinstance(r, pd.DataFrame) for r in results)
    assert all(not r.empty for r in results)


def test_thread_vs_process_execution(simple_mol):
    """Test both threading and multiprocessing execution."""
    # Test with threading
    pb_thread = ParallelPoseBusters(
        config="mol",
        n_workers=2,
        use_threading=True
    )
    result_thread = pb_thread.bust(simple_mol)

    # Test with multiprocessing
    pb_process = ParallelPoseBusters(
        config="mol",
        n_workers=2,
        use_threading=False
    )
    result_process = pb_process.bust(simple_mol)
    # Results should be equivalent
    assert result_thread.columns.equals(result_process.columns)
    assert result_thread.shape == result_process.shape


def test_cache_effectiveness(simple_mol):
    """Test that caching improves performance."""
    pb = ParallelPoseBusters(
        config="mol",
        cache_size=100
    )
    # First run should cache results
    result1 = pb.bust(simple_mol)
    # Second run should use cache
    result2 = pb.bust(simple_mol)
    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)


@pytest.mark.parametrize("n_workers", [1, 2, 4])
def test_different_worker_counts(simple_mol, n_workers):
    """Test performance with different numbers of workers."""
    pb = ParallelPoseBusters(
        config="mol",
        n_workers=n_workers
    )
    result = pb.bust(simple_mol)
    assert not result.empty


def test_error_handling(shared_datadir):
    """Test handling of invalid molecules and errors."""
    pb = ParallelPoseBusters(config="mol")

    # Test with invalid molecule
    invalid_mol = None
    with pytest.warns(UserWarning):
        result = pb.bust(invalid_mol)
        assert result is not None
