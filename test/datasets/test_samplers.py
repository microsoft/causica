import itertools

from causica.datasets.samplers import SubsetBatchSampler


def test_subset_batch_sampler_ordered():
    sampler = SubsetBatchSampler([2, 3, 5], 2, shuffle=False)
    samples = list(sampler)
    assert samples == [[0, 1], [2, 3], [4], [5, 6], [7, 8], [9]]
    assert len(samples) == len(sampler)


def test_subset_batch_sampler_ordered_drop_last():
    sampler = SubsetBatchSampler([2, 3, 5], 2, shuffle=False, drop_last=True)
    samples = list(sampler)
    assert samples == [[0, 1], [2, 3], [5, 6], [7, 8]]
    assert len(samples) == len(sampler)


def test_subset_batch_sampler_shuffled():
    sampler = SubsetBatchSampler([2, 3, 5], 2, shuffle=True)
    samples = list(sampler)
    sorted_samples = list(sorted(itertools.chain.from_iterable(samples)))
    assert sorted_samples == list(range(10))
    assert len(samples) == len(sampler)


def test_subset_batch_sampler_shuffled_drop_last():
    sampler = SubsetBatchSampler([2, 3, 5], 2, shuffle=True, drop_last=True)
    samples = list(sampler)
    concat_samples = list(itertools.chain.from_iterable(samples))
    assert len(concat_samples) == 8
    assert len(samples) == len(sampler)
    # First dataset is a single batch. The other two will lose a random sample each.
    assert 0 in concat_samples and 1 in concat_samples
