import argparse
import codecs
import json
import logging
import os
import sys

import numpy as np
import six
from six.moves import reload_module

from bleu.bleu import Bleu
from rouge.rouge import Rouge
from tokenizer.ptbtokenizer import PTBTokenizer

if six.PY2:
    reload_module(sys)
    sys.setdefaultencoding("utf-8")

stop_words = {"did", "have", "ourselves", "hers", "between", "yourself",
              "but", "again", "there", "about", "once", "during", "out", "very",
              "having", "with", "they", "own", "an", "be", "some", "for", "do", "its",
              "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s",
              "am", "or", "as", "from", "him", "each", "the", "themselves", "until", "below",
              "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were",
              "her", "more", "himself", "this", "down", "should", "our", "their", "while",
              "above", "both", "up", "to", "ours", "had", "she", "all", "no", "at", "any",
              "before", "them", "same", "and", "been", "have", "in", "will", "on", "does",
              "yourselves", "then", "that", "because", "over", "so", "can", "not", "now", "under",
              "he", "you", "herself", "has", "just", "too", "only", "myself", "those", "i", "after",
              "few", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "further",
              "was", "here", "than"}

question_words_global = {'what', 'which', 'why', 'who', 'whom', 'whose', 'where', 'when', 'how',
                         'What', 'Which', 'Why', 'Who', 'Whom', 'Whose', 'Where', 'When', 'How'}

_logger = logging.getLogger('answerability')

def remove_stopwords_and_NER_line(question, relevant_words=None, question_words=None):
    if relevant_words is None:

        question = question.split()
        if question_words is None:
           question_words = question_words_global

        temp_words = []
        for word in question_words:
            for i, w in enumerate(question):
                if w == word:
                    temp_words.append(w)
                    # If the question type is 'what' or 'which' the following word is generally associated with
                    # with the answer type. Thus it is important that it is considered a part of the question.
                    if i+1 < len(question) and (w.lower() == "what" or w.lower() == "which"):
                        temp_words.append(question[i+1])

        question_split = [item for item in question if item not in temp_words]
        ner_words = question_split
        temp_words = []

        for i in ner_words:
            if i[0].isupper() == False:
                if i not in stop_words :
                    temp_words.append(i)

        return " ".join(temp_words)
    else:
        question_words = question.split()
        temp_words = []
        for i in question_words:
            for j in relevant_words:
                if j.lower() in i:
                    temp_words.append(i)
        return " ".join(temp_words)

def NER_line(question):
    q_types = question_words_global
    question_words = question.split()
    if question_words[0].lower() in q_types:
        question_words = question_words[1:]

    temp_words = []
    for i in question_words:
        if i[0].isupper():
            temp_words.append(i)

    return " ".join(temp_words)


def get_stopwords(question):
    question_words = question.split()
    temp_words = []
    for i in question_words:
        if i.lower() in stop_words:
            temp_words.append(i.lower())

    return " ".join(temp_words)

def questiontype(question, questiontypes=None):

    if questiontypes == None:
        types = question_words_global
        question = question.strip()
        temp_words = []
        question = question.split()

        for word in types:
            for i, w in enumerate(question):
                if w == word:
                    temp_words.append(w)
                    if i+1 < len(question) and (w.lower() == "what" or w.lower() == "which"):
                        temp_words.append(question[i+1])

        return " ".join(temp_words)
    else:
        for i in questiontypes:
            if question.startswith(i + " "):
                return i
            else:
                return " "


def _get_json_format_qbleu(lines, output_path_prefix, relevant_words=None, questiontypes=None):
    if not os.path.exists(os.path.dirname(output_path_prefix)):
        os.makedirs(os.path.dirname(output_path_prefix))
    name = output_path_prefix + '_components'
    pred_sents_impwords = []
    pred_sents_qt = []
    pred_sents_ner = []
    pred_sents = []
    pred_sents_sw = []
    for line in lines:
        line_impwords = remove_stopwords_and_NER_line(line, relevant_words)
        line_ner = NER_line(line)
        line_sw = get_stopwords(line)
        line_qt = questiontype(line, questiontypes)
        pred_sents.append(line)
        pred_sents_impwords.append(line_impwords)
        pred_sents_ner.append(line_ner)
        pred_sents_qt.append(line_qt)
        pred_sents_sw.append(line_sw)

    ref_files = [os.path.join(name + "_impwords"), os.path.join(name + "_ner"), os.path.join(name + "_qt"), os.path.join(name + "_fluent"), os.path.join(name + "_sw")]

    data_pred_ner = []
    data_pred_qt = []
    data_pred_impwords = []
    data_pred = []
    data_pred_sw = []


    for id, s in enumerate(pred_sents_impwords):
        data_pred_impwords.append(dict(image_id=id, caption=s))
        data_pred_qt.append(dict(image_id=id, caption=pred_sents_qt[id]))
        data_pred_ner.append(dict(image_id=id, caption=pred_sents_ner[id]))
        data_pred.append(dict(image_id=id, caption=pred_sents[id]))
        data_pred_sw.append(dict(image_id=id, caption=pred_sents_sw[id]))

    with open(ref_files[0], 'w') as f:
        json.dump(data_pred_impwords, f, separators=(',', ':'))
    with open(ref_files[1], 'w') as f:
        json.dump(data_pred_ner, f, separators=(',', ':'))
    with open(ref_files[2], 'w') as f:
        json.dump(data_pred_qt, f, separators=(',', ':'))
    with open(ref_files[3], 'w') as f:
        json.dump(data_pred, f, separators=(',', ':'))
    with open(ref_files[4], 'w') as f:
        json.dump(data_pred_sw, f, separators=(',', ':'))

    return ref_files


def loadJsonToMap(json_file):
    with codecs.open(json_file, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    img_to_anns = {}
    for entry in data:
        if entry['image_id'] not in img_to_anns:
            img_to_anns[entry['image_id']] = []
        summary = dict(caption=entry['caption'], image_id=entry['caption'])
        img_to_anns[entry['image_id']].append(summary)
    return img_to_anns


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.keys()}


    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco[imgId]#.imgToAnns[imgId]
            res[imgId] = self.cocoRes[imgId]#.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
        self.setEvalImgs()
        return self.evalImgs
   
    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

                     
def compute_answerability_scores(all_scores, ner_weight, qt_weight, re_weight, d, output_dir, ngram_metric="Bleu_4",
                                 save_to_files=False):
    _logger.info("Number of samples: %s", len(all_scores))
    fluent_scores = [x[ngram_metric] for x in all_scores]
    imp_scores =  [x['imp'] for x in all_scores]
    qt_scores = [x['qt'] for x in all_scores]
    sw_scores = [x['sw'] for x in all_scores]
    ner_scores =  [x['ner'] for x in all_scores]

    new_scores = []

    for i in range(len(imp_scores)):
        answerability = re_weight*imp_scores[i] + ner_weight*ner_scores[i]  + \
            qt_weight*qt_scores[i] + (1-re_weight - ner_weight - qt_weight)*sw_scores[i]

        temp = d*answerability + (1-d)*fluent_scores[i]
        new_scores.append(temp)
        _logger.info("New Score: %.3f\nNER Score: %.3f\nRE Score: %.3f\nSW Score %.3f\nQT Score: %.3f",
                     temp, ner_scores[i], imp_scores[i], sw_scores[i], qt_scores[i])

    mean_answerability_score = np.mean(new_scores)
    mean_fluent_score = np.mean(fluent_scores)
    _logger.info("Mean Answerability Score Across Questions: %.3f\nN-gram Score: %.3f",
                 mean_answerability_score, mean_fluent_score)
    if save_to_files:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(os.path.join(output_dir, 'ngram_scores.txt'), fluent_scores)
        np.savetxt(os.path.join(output_dir, 'answerability_scores.txt'), new_scores)
    return mean_answerability_score, mean_fluent_score


def new_eval_metric(final_eval_perline_impwords, final_eval_perline_ner, final_eval_perline_qt, fluent_eval_perline, final_eval_perline_sw, new_scores):

    new_eval_per_line = []
    alpha = new_scores['alpha']
    beta = new_scores['beta']
    gamma = new_scores['gamma']
    theta = new_scores['theta']
    fluent_perline = 1 - alpha -beta -gamma-theta
    for i in range(len(final_eval_perline_impwords)):
        new_eval_alpha = alpha * final_eval_perline_impwords[i]['Bleu_1']
        new_eval_beta  = beta  * final_eval_perline_ner[i]['Bleu_1']
        new_eval_gamma = gamma * final_eval_perline_qt[i]['Bleu_1']
        new_eval_theta  = theta * final_eval_perline_sw[i]['Bleu_1']
        fluent_per_line = fluent_perline * fluent_eval_perline[i][sys.arg[7]]

        new_eval_per_line.append(new_eval_alpha + new_eval_beta + new_eval_gamma + new_eval_theta + fluent_per_line)

    return new_eval_per_line, np.mean(new_eval_per_line)


def get_answerability_scores(data_type,
                             delta,
                             hypothesis_lines,
                             ner_weight,
                             ngram_metric,
                             nist_meteor_scores_dir,
                             output_dir,
                             qt_weight,
                             re_weight,
                             references_lines,
                             save_to_files=False):
    if data_type == 'wikimovies':
        relevant_words = ['act', 'write', 'direct', 'describ', 'appear', 'star', 'genre', 'language', 'about', 'appear',
                          'cast']
        question_words = None
    else:
        relevant_words = None
        question_words = None
    filenames_1 = _get_json_format_qbleu(references_lines, os.path.join(output_dir, 'refs'),
                                         relevant_words, question_words)
    _logger.debug("Reference files written.")
    filenames_2 = _get_json_format_qbleu(hypothesis_lines, os.path.join(output_dir, 'hyps'),
                                         relevant_words, question_words)
    _logger.debug("Predicted files written.")
    final_eval = []
    final_eval_f = []
    true_sents = references_lines
    pred_sents = hypothesis_lines
    for file_1, file_2 in zip(filenames_1, filenames_2):
        coco = loadJsonToMap(file_1)
        cocoRes = loadJsonToMap(file_2)
        cocoEval_precision = COCOEvalCap(coco, cocoRes)
        cocoEval_recall = COCOEvalCap(cocoRes, coco)
        cocoEval_precision.params['image_id'] = cocoRes.keys()
        cocoEval_recall.params['image_id'] = cocoRes.keys()
        eval_per_line_p = cocoEval_precision.evaluate()
        eval_per_line_r = cocoEval_recall.evaluate()

        f_score = zip(eval_per_line_p, eval_per_line_r)
        temp_f = []
        for p, r in f_score:
            if (p['Bleu_1'] + r['Bleu_1'] == 0):
                temp_f.append(0)
                continue
            temp_f.append(2 * (p['Bleu_1'] * r['Bleu_1']) / (p['Bleu_1'] + r['Bleu_1']))

        final_eval_f.append(temp_f)
        final_eval.append(eval_per_line_p)
    fluent_scores = final_eval[3]
    if (nist_meteor_scores_dir == ""):
        nist_scores = [1] * len(pred_sents)
        meteor_scores = [1] * len(pred_sents)
    else:
        nist_scores = np.loadtxt(os.path.join(nist_meteor_scores_dir, "nist_scores"))
        meteor_scores = np.loadtxt(os.path.join(nist_meteor_scores_dir, "meteor_scores"))
    all_scores = zip(true_sents, pred_sents, final_eval_f[0], final_eval_f[1], final_eval_f[2], final_eval[3],
                     final_eval_f[4], nist_scores, meteor_scores)
    save_all = []
    for t, p, imp, ner, qt, fl, sw, nist, meteor in all_scores:
        save_all.append(
            {'true': t, 'pred': p, 'imp': imp, 'ner': ner, 'qt': qt, 'Bleu_1': fl['Bleu_1'], 'Bleu_2': fl['Bleu_2'],
             'Bleu_3': fl['Bleu_3'], 'Bleu_4': fl['Bleu_4'], 'Rouge_L': fl['ROUGE_L'], \
             'sw': sw, 'meteor': meteor, 'nist': nist})
    return compute_answerability_scores(save_all, ner_weight, qt_weight, re_weight, delta, output_dir, ngram_metric,
                                        save_to_files=save_to_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get the arguments')
    parser.add_argument('--data_type', dest='data_type', type=str,
                        help="Whether the data_type is [squad, wikimovies,vqa]. The relevant words in case of wikimovies is different.")
    parser.add_argument('--ref_file', dest='ref_file', type=str, help="Path to the reference question files")
    parser.add_argument('--hyp_file', dest='hyp_file', type=str, help="Path to the predicted question files")
    parser.add_argument('--ner_weight', dest='ner_weight', type=float, help="Weight to be given to NEs")
    parser.add_argument('--qt_weight', dest='qt_weight', type=float, help="Weight to be given to Question types")
    parser.add_argument('--re_weight', dest='re_weight', type=float, help="Weight to be given to Relevant words")
    parser.add_argument('--delta', dest='delta', type=float, help="Weight to be given to answerability scores")
    parser.add_argument('--output_dir', dest='output_dir', type=str,
                        help="Path to directory to store the scores per line, and auxilariy files")
    parser.add_argument('--ngram_metric', dest='ngram_metric', type=str,
                        help="N-gram metric that needs to be considered")
    parser.add_argument('--nist_meteor_scores_dir', dest="nist_meteor_scores_dir", type=str, default="",
                        help="Nist and Meteor needs to computed through different tools, provide the path to the precomputed scores")
    args = parser.parse_args()

    logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(filename)s::%(funcName)s\n%(message)s',
                        level=logging.INFO)

    with open(args.hyp_file, 'r') as f:
        hypothesis_lines = f.readlines()
    with open(args.ref_file, 'r') as f:
        references_lines = f.readlines()

    get_answerability_scores(args.data_type,
                             args.delta,
                             hypothesis_lines,
                             args.ner_weight,
                             args.ngram_metric,
                             args.nist_meteor_scores_dir,
                             args.output_dir,
                             args.qt_weight,
                             args.re_weight,
                             references_lines,
                             save_to_files=True)
