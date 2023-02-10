from causica.training.training_callbacks import AverageMetricTracker


def test_average_metric_tracker():
    length = 7
    avg_metric_tracker = AverageMetricTracker(averaging_period=length)
    values = list(reversed(range(length + 1)))
    for index in range(length):
        avg_metric_tracker.step(values[index])
        assert avg_metric_tracker.min_value == sum(values[: index + 1]) / (index + 1)
    # now we should overrun the buffer
    avg_metric_tracker.step(values[-1])
    assert avg_metric_tracker.min_value == sum(values[1:]) / length

    # check reset
    avg_metric_tracker.reset()
    avg_metric_tracker.step(values[length // 2])
    assert avg_metric_tracker.min_value == values[length // 2]
