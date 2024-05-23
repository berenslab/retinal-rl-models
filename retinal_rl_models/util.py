def assert_list(list_candidate, len_list, dtype=int):
    if isinstance(list_candidate, dtype):
        _list = [list_candidate] * len_list
    else:
        assert len(list_candidate) == len_list
        _list = list_candidate
    return _list
