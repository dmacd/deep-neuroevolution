import pytest
from multiplai.evals.nlu import data
from multiplai.evals.nlu import embedding as emb


def test_build_training_pairs(train_data, embedding):
    example = {
        'data':
            [{'text': 'What will the weather be, '},  # inject comma
             {'text': 'this year,', 'entity': 'timeRange'}, # inject comma
             {'text': ' in '}, {
                 # inject . for abbrev
                 'text': 'Horseshoe Lake St. Fish and Wildlife Area',
                 'entity': 'geographic_poi'}, {'text': '?'}]}

    all_text, all_entities = data.get_train_data_text_and_entities(train_data)

    pair = data.example_to_training_pair(
        example,
        data.get_word_to_idx(embedding),
        data.get_entity_to_idx(all_entities))

    assert pair == [(10, 0), (12, 0), (2, 0), (3, 0), (4, 0), (13, 0), (48, 9),
                    (154, 9), (13, 9), (1, 0), (1439, 6), (81, 6), (829, 6),
                    (23, 6), (741, 6), (35, 6), (60, 6), (100, 6), (8, 0)]


# todo: test pairs_from_data

def test_pairs_from_data():

    pass