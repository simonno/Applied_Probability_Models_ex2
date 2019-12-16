import math
import sys
from datetime import datetime
from decimal import Decimal
from random import randint

from HeldOutModel import HeldOutModel
from utils import words_splitter, get_words_list

X = 300000


#
# def p_ho(event, training, held_out, unseen_events):
#     r = training.get(event, 0)
#     if r == 0:
#         r_events = unseen_events
#
#     else:
#         r_events = [event for event, count in training.items() if count == r]
#
#     tr = sum([count for event, count in held_out.items() if event in r_events])
#     Nr = len(r_events)
#     ho_size = sum(held_out.values())
#     return tr / (ho_size * Nr)


def p_lid(event, words_dict, lamb=0.0):
    return (words_dict.get(event, 0) + lamb) / (sum(words_dict.values()) + lamb * X)


def init(outputs, development_set_filename, test_set_filename, input_word,
         output_filename):
    outputs.append(development_set_filename)
    outputs.append(test_set_filename)
    outputs.append(input_word)
    outputs.append(output_filename)
    outputs.append(X)
    outputs.append(1 / X)


def development_set_preprocessing(outputs, total):
    outputs.append(total)


def get_min_perplexity(training, validation):
    min_perplexity = float(math.inf)
    lamb = 0
    stop = Decimal('0.8')
    step = Decimal('0.01')
    curr_lamb = Decimal('0.0')
    while curr_lamb <= stop:
        curr_perplexity = perplexity(training, validation, curr_lamb)
        if curr_perplexity < min_perplexity:
            min_perplexity = curr_perplexity
            lamb = curr_lamb
        curr_lamb += step

    return min_perplexity, lamb


def perplexity(training, validation, lamb=0.0):
    start_perp = datetime.now()
    sum_p = 0
    for event, count in validation.items():
        prob = p_lid(event, training, lamb)
        if prob == 0:
            sum_p = float(-math.inf)
            break
        else:
            sum_p += count * math.log(prob)
    prep = math.pow(2, -sum_p / sum(validation.values()))
    print('perplexity {0} - lambda {1} running time: {2}'.format(prep, lamb, datetime.now() - start_perp))
    return prep


def debug_lidstone(training):
    lamb = randint(0, 200) * 0.01
    return sum([p_lid(event, training, lamb) for event in training.keys()]) + p_lid('unseen-word', training, lamb) * (
            X - len(training.keys()))


def lidstone_model_training(outputs, input_word, words_list):
    start_lid = datetime.now()
    training, validation = words_splitter(words_list)
    outputs.append(sum(validation.values()))
    outputs.append(sum(training.values()))
    outputs.append(len(training.keys()))
    outputs.append(training.get(input_word, 0))
    outputs.append(p_lid(input_word, training))
    outputs.append(p_lid('unseen-word', training))
    outputs.append(p_lid(input_word, training, 0.1))
    outputs.append(p_lid('unseen-word', training, 0.1))
    outputs.append(perplexity(training, validation, 0.01))
    outputs.append(perplexity(training, validation, 0.1))
    outputs.append(perplexity(training, validation, 1))
    min_perplexity, lamb = get_min_perplexity(training, validation)
    outputs.append(lamb)
    outputs.append(min_perplexity)
    print('Lidstone debug value: {0}'.format(debug_lidstone(training)))
    print('Lidstone running time: {0}'.format(datetime.now() - start_lid))
    return lamb


def write_outputs(outputs, output_filename):
    with open(output_filename, 'w') as file:
        file.write('#Students\tNoam Simon\tChen Eliyahou\t208388850\t312490675\n')
        for index in range(len(outputs)):
            file.write('#Output{0}\t{1}\n'.format(index + 1, outputs[index]))


# def debug_held_out(training, held_out, unseen_events):
#     sum_p = p_ho('unseen-word', training, held_out, unseen_events) * len(unseen_events)
#     for event in training.keys():
#         start_p_ho = datetime.now()
#         sum_p += p_ho(event, training, held_out, unseen_events)
#         print('Held Out Debug: event {0} ,running time: {1}'.format(event, datetime.now() - start_p_ho))
#     return sum_p


def held_out_model_training(outputs, input_word, words):
    start_ho = datetime.now()
    held_out_model = HeldOutModel(words)
    outputs.append(held_out_model.get_training_size())
    outputs.append(held_out_model.get_held_out_size())
    outputs.append(held_out_model.get_prob(input_word))
    outputs.append(held_out_model.get_prob('unseen-word'))

    print('Held-Out debug value: {0}'.format(held_out_model.debug()))
    print('Held-Out running time: {0}'.format(datetime.now() - start_ho))


def models_evaluation_on_test(outputs, test, lamb):
    test_dict, _ = words_splitter(test, 1.0)
    outputs.append(len(test))
    outputs.append(perplexity())


def main(development_set_filename, test_set_filename, input_word, output_filename):
    outputs = list()
    dev = get_words_list(development_set_filename)
    test = get_words_list(test_set_filename)
    init(outputs, development_set_filename, test_set_filename, input_word, output_filename)
    development_set_preprocessing(outputs, len(dev))
    # lamb = lidstone_model_training(outputs, input_word, dev)
    held_out_model_training(outputs, input_word, dev)
    # models_evaluation_on_test(outputs, test, lamb)

    write_outputs(outputs, output_filename)


if __name__ == "__main__":
    start = datetime.now()
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print('Total running time: {0}'.format(datetime.now() - start))
