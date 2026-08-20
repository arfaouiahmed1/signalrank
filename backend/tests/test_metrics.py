from backend.app.evaluation.metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k

def test_precision_recall():
    retrieved = [1,2,3,4,5]
    relevant = {2,4,6}
    assert precision_at_k(retrieved, relevant, 5) == 0.4
    assert recall_at_k(retrieved, relevant, 5) == 2/3
    assert abs(mrr(retrieved, relevant) - 0.5) < 1e-9

def test_ndcg():
    # graded: job 1 rel 2 should be top
    qrels = {1:2, 2:1, 3:0, 4:1}
    # perfect order
    assert abs(ndcg_at_k([1,2,4,3], qrels, 4) - 1.0) < 1e-9
    # worst order
    assert ndcg_at_k([3,4,2,1], qrels, 4) < 0.6

def test_mrr_no_hit():
    assert mrr([1,2,3], {9}) == 0.0
