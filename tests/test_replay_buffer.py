"""Tests for memory buffer"""

import pytest
import numpy as np
from supercopter.replay_buffer import Record, ReplayBuffer


def random_record():
    """Generate a random state"""
    return Record(start_state=np.random.rand(6*3), action=np.random.rand(4),
                  reward=np.random.rand(), end_state=np.random.rand(6*3))


@pytest.fixture
def empty_rb():
    """Return an empty ReplayBuffer"""
    return ReplayBuffer()

@pytest.fixture
def full_rb():
    """return a full ReplayBuffer"""
    rb = ReplayBuffer(maxlen=10)
    for _ in range(10):
        rb.add(random_record())
    return rb


def test_named_tuple_class():
    """Tests if a named tuple class has been defined for memory buffer"""
    random_record()


def test_create_replay_buffer():
    """We can instantiate a replay buffer"""
    ReplayBuffer()
    ReplayBuffer(maxlen=2000)


def test_record_only(empty_rb):
    """We can add records only"""
    empty_rb.add(random_record())


@pytest.mark.parametrize('added_obj', [
    42, "Question?", [42, None]
])
def test_raises_if_add_random_type(added_obj, empty_rb):
    """Should raise TypeError if object added != Record"""
    with pytest.raises(TypeError):
        empty_rb.add(added_obj)


def test_query_size(empty_rb):
    """Get the current size of ReplayBuffer object"""
    assert len(empty_rb) == 0
    empty_rb.add(random_record())
    assert len(empty_rb) == 1


def test_max_len_enforced(full_rb):
    """Check if we discard oldest records when rb is full"""
    size_rb = len(full_rb)
    full_rb.add(random_record())
    assert size_rb == len(full_rb)


def test_sample(full_rb):
    """Get sample of what's inside the buffer"""
    sample = full_rb.sample(5)
    assert isinstance(sample, list)
    assert isinstance(sample[0], Record)


@pytest.mark.parametrize('nb_elem', [0, 5, 100])
def test_sample_size_higher_than_rb_size(nb_elem, empty_rb):
    """Return all elements (in random order) but not more"""
    for _ in range(nb_elem):
        empty_rb.add(random_record())
    assert len(empty_rb.sample(100)) == nb_elem


def test_sample_raises_if_size_higher_than_maxlen():
    """Raises ValueError if sample's size > rb's maxlen"""
    rb = ReplayBuffer(maxlen=10)
    with pytest.raises(ValueError):
        rb.sample(1000)
