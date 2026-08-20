from backend.app.retrieval.hybrid import rrf_fuse

def test_rrf_fuse():
    a = [{"id":1,"rank":1},{"id":2,"rank":2}]
    b = [{"id":2,"rank":1},{"id":1,"rank":2}]
    fused = rrf_fuse([a,b], k=60)
    # Both 1 and 2 appear twice; tie broken by first list weight
    ids = [x["id"] for x in fused]
    assert set(ids) == {1,2}
    assert fused[0]["rrf_score"] > 0

def test_rrf_weights():
    a = [{"id":1,"rank":1}]
    b = [{"id":2,"rank":1}]
    fused = rrf_fuse([a,b], k=60, weight=[2.0,1.0])
    assert fused[0]["id"] == 1
