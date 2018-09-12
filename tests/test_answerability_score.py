import os

from numpy.testing import assert_almost_equal

from answerability_score import get_answerability_scores

_output_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '../../auxiliary_output'))


def test_get_answerability_scores_Bleu_3():
    mean_answerability_score, mean_fluent_score = get_answerability_scores(hypotheses=["Is this a great question?"],
                                                                           references=["Is this a good question?"],
                                                                           ngram_metric='Bleu_3',
                                                                           delta=0.7,
                                                                           ner_weight=0.6,
                                                                           qt_weight=0.2,
                                                                           re_weight=0.1,
                                                                           output_dir=_output_dir)
    assert_almost_equal(mean_answerability_score, 0.678, decimal=3)
    assert_almost_equal(mean_fluent_score, 0.511, decimal=3)


def test_get_answerability_scores_Rouge_L():
    mean_answerability_score, mean_fluent_score = get_answerability_scores(hypotheses=["Is this a great question?"],
                                                                           references=["Is this a good question?"],
                                                                           ngram_metric='ROUGE_L',
                                                                           delta=0.7,
                                                                           ner_weight=0.6,
                                                                           qt_weight=0.2,
                                                                           re_weight=0.1,
                                                                           output_dir=_output_dir)
    assert_almost_equal(mean_answerability_score, 0.765, decimal=3)
    assert_almost_equal(mean_fluent_score, 0.8, decimal=3)
