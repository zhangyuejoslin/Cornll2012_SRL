

def labels_to_output(labels):
    in_span = False
    output = []
    for i, label in enumerate(labels):

        out_label = "*"
        if label.startswith('B'): # start span
            in_span = True
            formatted_span_name = label[2:].replace('ARG', 'A')
            out_label = f'({formatted_span_name}' + out_label

        if i < len(labels) -1 and in_span and not labels[i+1].startswith('I'):
            in_span = False
            out_label = out_label + ')'

        if in_span and i == len(labels) - 1:
            out_label = out_label +  ')'

        output.append(out_label)
    return output

def save_predictions(input_sentences, predicate_ids, predictions, output_file):
    """
    input_sentences: 2D list [sentences x length of sentences]
    predicate_ids: List of ids
    predictions: 2D list [predictions x length of predictions]
    """
    sentences = []

    last_sentence = 'n/a'
    for words, labels, pred_id in zip(input_sentences, predictions, predicate_ids):
        sentence = ','.join(words)
        if last_sentence != sentence:
            last_sentence = sentence
            new_q = []
            for _ in range(len(words)):
                new_q.append(list(['-']))
            sentences.append(new_q)

        # add current predicate
        sentences[-1][pred_id][0] = words[pred_id]
        # add current predicate spans
        for i, output_label in enumerate(labels_to_output(labels)):
            sentences[-1][i].append(output_label)

    with open(output_file, 'w') as output_file:
        output_file.write('\n\n'.join(['\n'.join([' '.join(line) for line in q]) for q in sentences]))


save_predictions([['What','is', 'your', 'deal'],['What','is', 'your', 'deal'],['What', 'your', 'is', 'deal']], [1,2,3],[['B-ARG1','B-V','O','O'],['B-ARG1','B-V','O','O'],['B-ARG1','B-V','O','O']], 'output.txt')