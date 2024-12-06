def box_ids2seq_id(box_ids):
    box_ids_copy = box_ids.copy()
    box_ids_sorted = sorted(box_ids_copy, reverse=True)
    box_ids_str = ''.join([str(box_id) for box_id in box_ids_sorted])

    return int(box_ids_str)

def test(box_ids):
    sampling_result = dict()
    
    seq_ids = [list(map(box_ids2seq_id, box_ids_)) for box_ids_ in box_ids]
    print("seq_ids ",seq_ids)
    seq_ids_per_image = []
    start_id = 0
    for seq_ids_ in seq_ids:
        print(seq_ids_)
        seq_ids_per_image.extend([box_id + start_id for box_id in seq_ids_])
        start_id += (max(seq_ids_) + 1)
        print(start_id)
    # sampling_result.set_field(name='seq_ids', value=seq_ids_per_image,
    #                           field_type='metainfo', dtype=None)
    
    print(seq_ids_per_image)
    
if __name__ == '__main__':
    box_ids = [[2, 5, 7, 1, 4], [4, 2, 1, 7, 5]], [[8, 6, 4, 7, 0], [0, 8, 6, 7, 4]], [[5, 7, 4, 2], [2, 4, 5, 7]], [[4, 5, 3], [3, 5, 4]]
    
    test(box_ids)