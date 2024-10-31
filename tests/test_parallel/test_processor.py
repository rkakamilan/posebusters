from rdkit.Chem import MolFromSmiles
from posebusters.parallel.processor import MoleculeProcessor


def test_processor_initialization():
    """Test processor initialization."""
    processor = MoleculeProcessor()
    assert processor is not None


def test_empty_results_generation():
    """Test generation of empty results."""
    processor = MoleculeProcessor()
    module_configs = []
    empty_results = processor.get_empty_results(module_configs)
    assert isinstance(empty_results, dict)


def test_module_output_retrieval():
    """Test retrieval of module outputs."""
    processor = MoleculeProcessor()
    module_configs = [("test", "test_func", lambda x: x, ["output1"])]
    outputs = processor.get_module_outputs(module_configs, "test_func")
    assert outputs == ["output1"]