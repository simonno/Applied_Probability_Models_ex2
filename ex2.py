import sys
from datetime import datetime
from random import randint

from FrequencyTable import FrequencyTable
from HeldOutModel import HeldOutModel
from LidstoneModel import LidstoneModel
from utils import get_words_list

X = 300000


def init(outputs, development_set_filename, test_set_filename, input_word, output_filename):
    outputs.append(development_set_filename)
    outputs.append(test_set_filename)
    outputs.append(input_word)
    outputs.append(output_filename)
    outputs.append(X)
    outputs.append(1 / X)


def development_set_preprocessing(outputs, total):
    outputs.append(total)


def lidstone_model_training(outputs, input_word, dev_words_list, test_words_list):
    start_lid = datetime.now()

    lidstone_model = LidstoneModel(dev_words_list, test_words_list, X)
    outputs.append(lidstone_model.get_validation_size())
    outputs.append(lidstone_model.get_training_size())
    outputs.append(lidstone_model.get_training_different_events())
    outputs.append(lidstone_model.get_event_appearance(input_word))
    outputs.append(lidstone_model.get_prob(input_word, 0.0))
    outputs.append(lidstone_model.get_prob('unseen-word', 0.0))
    outputs.append(lidstone_model.get_prob(input_word, 0.01))
    outputs.append(lidstone_model.get_prob('unseen-word', 0.1))
    outputs.append(lidstone_model.get_validation_perplexity(0.01))
    outputs.append(lidstone_model.get_validation_perplexity(0.1))
    outputs.append(lidstone_model.get_validation_perplexity(1))
    min_perplexity, best_lambda = lidstone_model.get_min_perplexity(0.0, 2.0, 0.01)
    outputs.append(best_lambda)
    outputs.append(min_perplexity)
    print('Lidstone debug value: {:.5f}'.format(lidstone_model.debug(randint(0, 200) * 0.01)))
    print('Lidstone running time: {0}'.format(datetime.now() - start_lid))

    return lidstone_model, best_lambda


def write_outputs(outputs, output_filename):
    with open(output_filename, 'w') as file:
        file.write('#Students\tNoam Simon\tChen Eliyahou\t208388850\t312490675\n')
        for index in range(len(outputs)):
            file.write('#Output{0}\t{1}\n'.format(index + 1, outputs[index]))


def held_out_model_training(outputs, input_word, dev_words_list, test_words_list):
    start_ho = datetime.now()

    held_out_model = HeldOutModel(dev_words_list, test_words_list, X)
    outputs.append(held_out_model.get_training_size())
    outputs.append(held_out_model.get_held_out_size())
    outputs.append(held_out_model.get_prob(input_word))
    outputs.append(held_out_model.get_prob('unseen-word'))

    print('Held-Out debug value: {:.5f}'.format(held_out_model.debug()))
    print('Held-Out running time: {0}'.format(datetime.now() - start_ho))
    return held_out_model


def models_evaluation_on_test(outputs, held_out_model, lidstone_model, best_lambda):
    start_evaluation = datetime.now()

    outputs.append(lidstone_model.get_test_size())
    lidstone_perplexity = lidstone_model.get_test_perplexity(best_lambda)
    held_out_perplexity = held_out_model.get_test_perplexity()
    outputs.append(lidstone_perplexity)
    outputs.append(held_out_perplexity)
    outputs.append('L' if lidstone_perplexity < held_out_perplexity else 'H')
    outputs.append(FrequencyTable(held_out_model, lidstone_model, best_lambda, 10).to_string())

    print('Model evaluation running time: {0}'.format(datetime.now() - start_evaluation))


def main(development_set_filename, test_set_filename, input_word, output_filename):
    outputs = list()

    dev = get_words_list(development_set_filename)
    test = get_words_list(test_set_filename)
    init(outputs, development_set_filename, test_set_filename, input_word, output_filename)
    development_set_preprocessing(outputs, len(dev))
    lidstone_model, best_lambda = lidstone_model_training(outputs, input_word, dev, test)
    held_out_model = held_out_model_training(outputs, input_word, dev, test)
    models_evaluation_on_test(outputs, held_out_model, lidstone_model, best_lambda)

    write_outputs(outputs, output_filename)


if __name__ == "__main__":
    start = datetime.now()
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print('Total running time: {0}'.format(datetime.now() - start))
