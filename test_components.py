__author__ = 'tylin'
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
#from bleu.bleu import cBleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
#from cider.cider import Cider
import json
import sys
import os
import codecs
import numpy as np
reload(sys)
import pickle
import scipy.stats as s
sys.setdefaultencoding("utf-8")

stop_words = ["did", "have", "ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "over", "so", "can", "not", "now", "under", "he", "you", "herself", "has", "just",  "too", "only", "myself",  "those", "i", "after", "few", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "further", "was", "here", "than"]
 
def remove_stopwords_and_NER_line(question, relevant_words=None, question_words = None):

    if relevant_words == None:

        question = question.split()
        stop_words = ["did", "have", "ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "over", "so", "can", "not", "now", "under", "he", "you", "herself", "has", "just",  "too", "only", "myself",  "those", "i", "after", "few", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "further", "was", "here", "than"]
        if question_words == None:
            question_words = ['what','which','who','whom','whose','where','when','how','what','Which', 'Why', 'Who', 'Whom', 'Whose', 'Where', 'When', 'How']

        temp_words = []
        for word in question_words:
            for i, w in enumerate(question):
                if w == word:
                    temp_words.append(w)
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

        #print (" ".join(temp_words))
        return " ".join(temp_words)

def NER_line(question):
    print question
    q_types = ['what','which','who','whom','whose','where','when','how','what','Which', 'Why', 'Who', 'Whom', 'Whose', 'Where', 'When', 'How']
    question_words = question.split()
    if question_words[0].lower() in q_types:
        question_words = question_words[1:]

    temp_words = []
    for i in question_words:
        if i[0].isupper() == True:
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
        types = ['what', 'which', 'why', 'who', 'whom', 'whose', 'where', 'when', 'how', 'what', 'Which', 'Why', 'Who', 'Whom', 'Whose', 'Where', 'When', 'How']
        question = question.strip()
        temp_words = []
        # Remove the word appearing after what or which
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

def get_components(hypotheis, reference):

    qt_h, sw_h, ner_h, rel_h = questiontype(hypotheis), get_stopwords(hypotheis), NER_line(hypotheis), remove_stopwords_and_NER_line(hypotheis)
    qt_r, sw_r, ner_r, rel_r = questiontype(reference), get_stopwords(reference), NER_line(reference), remove_stopwords_and_NER_line(reference)

    precision = []
    recall = []

    qt_in = [i for i in qt_h if i in qt_r]
    sw_in = [i for i in sw_h if i in sw_r]
    ner_in = [i for i in ner_h if i in ner_r]
    rel_in = [i for i in rel_h if i in rel_r]

    
def _get_json_format_qbleu(o, name, relevant_words=None, questiontypes=None, num_points = 300):
     #p_file = codecs.open(o, "r", encoding="utf-8", errors="ignore")
     p_file = open(o, "r")
     pred_sents_impwords = []
     pred_sents_qt = []
     pred_sents_ner = []
     pred_sents = []
     pred_sents_sw = []
     count = 1
     p_file = p_file.readlines()
     print (len(p_file))
     for line in p_file:
        if count >  num_points:
            break
        count += 1
        line_impwords = remove_stopwords_and_NER_line(line, relevant_words)
        line_ner = NER_line(line)
        line_sw = get_stopwords(line)
        line_qt = questiontype(line, questiontypes)
        pred_sents.append(line)
        pred_sents_impwords.append(line_impwords)
        pred_sents_ner.append(line_ner)
        pred_sents_qt.append(line_qt)
        pred_sents_sw.append(line_sw)

        #print (line_impwords, line_ner, line_sw, line_qt)

     ref_files = [os.path.join(name + "_impwords"), os.path.join(name + "_ner"), os.path.join(name + "_qt"), os.path.join(name + "_fluent"), os.path.join(name + "_sw")]

     data_pred_ner = []
     data_pred_qt = []
     data_pred_impwords = []
     data_pred = []
     data_pred_sw = []


     for id, s in enumerate(pred_sents_impwords):
        line = {}
        line['image_id'] = id
        line['caption'] = s
        data_pred_impwords.append(line)

        line = {}
        line['image_id'] = id
        line['caption'] = pred_sents_qt[id]
        data_pred_qt.append(line)

        line = {}
        line['image_id'] = id
        line['caption'] = pred_sents_ner[id]
        data_pred_ner.append(line)

        line = {}
        line['image_id'] = id
        line['caption'] = pred_sents[id]
        data_pred.append(line)

        line = {}
        line['image_id'] = id 
        line['caption'] = pred_sents_sw[id]
        data_pred_sw.append(line)

     json.dump(data_pred_impwords, open(ref_files[0], "w"))
     json.dump(data_pred_ner, open(ref_files[1], "w"))
     json.dump(data_pred_qt, open(ref_files[2], "w"))
     json.dump(data_pred, open(ref_files[3], "w"))
     json.dump(data_pred_sw, open(ref_files[4], "w"))

     return ref_files


def loadJsonToMap(json_file):
    data = json.load(codecs.open(json_file, "r", encoding="utf-8", errors="ignore"))
    #data = json.load(open(json_file, "r"))
    imgToAnns = {}
    for entry in data:
    #print entry['image_id'],entry['caption']
        if entry['image_id'] not in imgToAnns.keys():
                imgToAnns[entry['image_id']] = []
        summary = {}
        summary['caption'] = entry['caption']
        summary['image_id'] = entry['caption']
        imgToAnns[entry['image_id']].append(summary)
    return imgToAnns

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
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            #(cBleu(4), ["cBleu_1", "cBleu_2", "cBleu_3", "cBleu_4"]),
            #(Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L")
            #(Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print "%s: %0.3f"%(m, sc)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print "%s: %0.3f"%(method, score)
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
    

def new_eval_metric_grid(final_eval_perline_impwords, final_eval_perline_ner, final_eval_perline_qt, fluent_eval_perline, final_eval_perline_sw, ref_scores):

    print (final_eval_perline_impwords)
    print(final_eval_perline_ner)
    print(final_eval_perline_qt)
    print(fluent_eval_perline)
    w_re = np.arange(0.01,1, 0.05)
    w_ner = np.arange(0.01,1,0.05)
    w_qt = np.arange(0.01,1,0.05)
    w_sw = np.arange(0.01,1,0.05)

    d= np.arange(0.01,1,0.02)
    fluent_scores = []
    for i in fluent_eval_perline:
        fluent_scores.append(i[sys.argv[7]])

    new_scores = {}
    loss = 0 
    best_score = -1 * s.pearsonr(ref_scores, fluent_scores)[0]
    ind = 0
    for a in w_re:
        #print a
        for b in w_ner:
            if ((a + b )> 1):
                continue
            for g in w_qt:
                if ((a + b + g ) > 1):
                    continue    
                for t in d:
                    new_eval_per_line = []
                    for i in range(len(final_eval_perline_impwords)):
                        answerability = a*final_eval_perline_impwords[i] + b*final_eval_perline_ner[i]  + \
                                g*final_eval_perline_qt[i] + (1-a-b-g)*final_eval_perline_sw[i]
                        new_eval_per_line.append(t*answerability + (1-t)*fluent_scores[i])

                    #loss = np.mean((new_eval_per_line - ref_scores)* (new_eval_per_line - ref_scores))
                    loss = -1*s.pearsonr(new_eval_per_line, ref_scores)[0]
                        #print (loss)
                    if loss < best_score:
                        best_score = loss
                        print ("Current best score is", best_score, a, b,g,t)
                        new_scores = {"eval_list" : new_eval_per_line, "eval": np.mean(new_eval_per_line), "w_re": a, "w_ner": b, "w_qt":g,"d":d}

                    else:
                        continue

    print ("Parameters", new_scores['w_re'], new_scores['w_ner'], new_scores['w_qt'],new_scores['d'])
    pickle.dump(new_scores, open("best_alpha_beta_gamma.pkl", "w"))
    return new_scores
    
if __name__ == '__main__':
    # impwords, ner, qt, fluent 
    #relevant_words =  ['act', 'write', 'direct', 'describ', 'appear', 'star', 'genre', 'language', 'about','appear','cast']
    #question_words = None

    relevant_words = None
    question_words = None

    #relevant_words = None
    #question_words =  ["What color", "What is", "What kind", "What are", "What type", "What does", "What time", "What sport", "What animal", "What brand", "Is the", "Is there", "How many", "Are", "Does", "Where", "Why", "Which", "Do", "Who"]

    filenames_1 = _get_json_format_qbleu(sys.argv[1], sys.argv[3], relevant_words, question_words, num_points=int(sys.argv[6]))
    print ("reference_files written")
    filenames_2 = _get_json_format_qbleu(sys.argv[2], sys.argv[4],relevant_words, question_words, num_points=int(sys.argv[6]))
    print ("predicted files written")
    ref_pred = zip(filenames_1, filenames_2)

    final_eval = []
    final_eval_f = []

    true_sents = open(sys.argv[1])
    true_sents = true_sents.readlines()

    pred_sents = open(sys.argv[2])
    pred_sents = pred_sents.readlines()
    for file_1, file_2 in ref_pred:
        coco = loadJsonToMap(file_1)
        cocoRes = loadJsonToMap(file_2)
        cocoEval_precision = COCOEvalCap(coco, cocoRes)
        cocoEval_recall    = COCOEvalCap(cocoRes,coco)
        cocoEval_precision.params['image_id'] = cocoRes.keys()
        cocoEval_recall.params['image_id'] = cocoRes.keys()
        eval_per_line_p = cocoEval_precision.evaluate()
        eval_per_line_r = cocoEval_recall.evaluate()

        f_score = zip(eval_per_line_p, eval_per_line_r)
        temp_f = []
        for p,r in f_score:
            if (p['Bleu_1']+r['Bleu_1'] == 0):
                temp_f.append(0)
                continue
            temp_f.append( 2*(p['Bleu_1']*r['Bleu_1'])/(p['Bleu_1']+r['Bleu_1']))
            #print ("Precision recall", p, r, temp_f[-1])

        final_eval_f.append(temp_f)
        final_eval.append(eval_per_line_p)

        ref_scores = np.loadtxt(sys.argv[5])
        ref_scores = ref_scores[:int(sys.argv[6])]
        #ref_scores = [int(i.strip()) for i in ref_scores]
    fluent_scores = final_eval[3]

    meteor_scores = np.loadtxt('../../squad_meteor')
    nist_scores  = np.loadtxt('../../squad_nist')

    all_scores = zip(true_sents, pred_sents, ref_scores, final_eval_f[0],final_eval_f[1],final_eval_f[2],final_eval[3],final_eval_f[4], [1]*1000, [1]*1000)#nist_scores, meteor_scores)

    save_all = []
    for t, p, r,imp,ner,qt,fl,sw,meteor,nist in all_scores:
        print (t, p)
        print ("ner score {}, re_score {} qt score{}".format(ner, imp, qt))
        save_all.append({'true': t, 'pred':p,'ref':r,'imp':imp,'ner':ner,'qt':qt,'fl_1':fl['Bleu_1'],'fl_2':fl['Bleu_2'],'fl_3':fl['Bleu_3'],'fl_4':fl['Bleu_4'],'fl_rouge':fl['ROUGE_L'], \
                        'sw':sw, 'meteor':meteor,'nist':nist})

    pickle.dump(save_all, open('all_metrics_quora.pkl','w'))


    #new_scores  = new_eval_metric_grid(final_eval_f[0], final_eval_f[1], final_eval_f[2], fluent_scores, final_eval_f[4], ref_scores)

    """
       filenames_1 = _get_json_format_qbleu(sys.argv[1], sys.argv[3], relevant_words, question_words, num_points=771)
       #print ("reference_files written")
       filenames_2 = _get_json_format_qbleu(sys.argv[2], sys.argv[4],relevant_words, question_words, num_points=771)
       #print ("predicted files written")
       ref_pred = zip(filenames_1, filenames_2)
       final_eval = []
       for file_1, file_2 in ref_pred:
           coco = loadJsonToMap(file_1)
           cocoRes = loadJsonToMap(file_2)
           cocoEval_precision = COCOEvalCap(coco, cocoRes)
           cocoEval_recall    = COCOEvalCap(cocoRes,coco)
           cocoEval_precision.params['image_id'] = cocoRes.keys()
           cocoEval_recall.params['image_id'] = cocoRes.keys()
           eval_per_line_p = cocoEval_precision.evaluate()
           eval_per_line_r = cocoEval_recall.evaluate()

           f_score = zip(eval_per_line_p, eval_per_line_r)
           temp_f = []
           for i,j in f_score:
                temp_f.append( p*r/(p+r))
                
            final_eval_f.append(temp_f)
            final_eval.append(eval_per_line)

       new_evals, total_eval = new_eval_metric(final_eval_f[0], final_eval_f[1], final_eval_f[2], fluent_scores, final_eval_f[4], new_scores)
       pickle.dump([new_evals, total_eval], open(sys.argv[3] + ".pkl","w"))
       np.savetxt(sys.argv[3] + "_scores", new_evals, delimiter="\n")
       print total_eval

def main():
    s = "Wakanda is ruled by which king ?"
    print ("Question", s)
    print ("question type", questiontype(s))
    print ("stop words", get_stopwords(s))
    print("NER", NER_line(s))
    print("Relevant words", remove_stopwords_and_NER_line(s))
    
    """
