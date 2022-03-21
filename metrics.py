import os
import config
import logging


def get_entities(seq):
    """
    Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def get_entities_name(entities_list, data):
    types = {}
    if any(isinstance(s, list) for s in data):
        data = [item for sublist in data for item in sublist + ['#']]
    # data1 = [y for x in data for y in x]
    for e in entities_list:
        if e[0] not in types:
            types[e[0]] = []
        s = ''
        for i in range(e[1], e[2] + 1):
            s = s + data[i]
        types[e[0]].append(s)

    return types


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    if prev_tag == 'E':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'M' and tag == 'B':
        chunk_end = True
    if prev_tag == 'M' and tag == 'S':
        chunk_end = True
    if prev_tag == 'M' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'S' and tag == 'M':
        chunk_start = True
    if prev_tag == 'O' and tag == 'M':
        chunk_start = True
    if prev_tag == 'E' and tag == 'M':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, mode='dev'):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    # print(pred_entities)

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    avg_score = {'p': p, 'r': r, 'f1': score}
    if mode == 'dev':
        return avg_score
    else:
        label_score = {}
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            label_score[label] = {'p': p_label, 'r': r_label, 'f1': score_label}
        return label_score, avg_score


def f1_score_overlapping(y_true, y_pred, mode='dev'):
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    # print(pred_entities)

    # nb_correct = len(true_entities & pred_entities)
    nb_correct = 0
    correct_entities = set()
    for p in pred_entities:
        for t in true_entities:
            if t not in correct_entities:
                if t[0] == p[0]:
                    if max(p[1], t[1]) <= min(p[2], t[2]):
                        correct_entities.add(t)
                        nb_correct = nb_correct + 1
                        break

    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    avg_score = {'p': p, 'r': r, 'f1': score}
    if mode == 'dev':
        return avg_score
    else:
        label_score = {}
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            # nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_correct_label = 0
            correct_entities_label = set()
            for p in pred_entities_label:
                for t in true_entities_label:
                    if t not in correct_entities_label:
                        if max(p[1], t[1]) <= min(p[2], t[2]):
                            correct_entities_label.add(t)
                            nb_correct_label = nb_correct_label + 1
                            break

            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            label_score[label] = {'p': p_label, 'r': r_label, 'f1': score_label}
        return label_score, avg_score


def bad_case(y_true, y_pred, data):
    if not os.path.exists(config.case_dir):
        os.system(r"echo test {}".format(config.case_dir))  # 调用系统命令行来创建文件
    output = open(config.case_dir, 'w', encoding='utf-8')
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        else:
            output.write("bad case " + str(idx) + ": \n")
            output.write("sentence: " + str(data[idx]) + "\n")
            output.write("golden label: " + str(t) + "\n")
            output.write("model pred: " + str(p) + "\n")
    logging.info("--------Bad Cases reserved !--------")


if __name__ == "__main__":
    y_t = [['O', 'O', 'O', 'B-灾害地点', 'M-灾害地点', 'E-灾害地点', 'O'], ['B-灾害类型', 'M-灾害类型', 'O']]
    y_p = [['O', 'O', 'B-灾害地点', 'M-灾害地点', 'E-灾害地点', 'B-灾害类型', 'E-灾害类型'], ['B-灾害类型', 'M-灾害类型', 'O']]
    sent = [['十', '一', '月', '中', '山', '路', '电'], ['周', '静', '说']]
    bad_case(y_t, y_p, sent)
    print(get_entities(y_p))
    entities = get_entities_name(get_entities(y_p), sent)
    print(entities)
