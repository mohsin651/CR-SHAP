import pytest

from experiment4r.core import partition


def test_exact_original_eligible_population_and_disjoint_remainder():
    screening=[{'image_id':str(i),'original_margin':1 if i<793 else -1} for i in range(1000)]
    population,remainder,digest=partition(screening,[str(i) for i in range(300)])
    assert len(population)==793 and len(remainder)==493 and len(digest)==64
    assert not {r['image_id'] for r in remainder}&{str(i) for i in range(300)}
    assert partition(list(reversed(screening)),[str(i) for i in range(300)])[2]==digest
    with pytest.raises(ValueError): partition(screening,[str(i) for i in range(300,601)])
