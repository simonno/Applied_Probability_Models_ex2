import sys

# outputs_dict[

V_size = 300000


def p(event, words_dict, lamb=0.0):
    return (words_dict.get(event, 0) + lamb) / (sum(words_dict.values()) + lamb * (len(words_dict.keys())))


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines


def get_words_list_and_dict(filename):
    words_list = list()
    words_dict = dict()
    lines = read_lines(filename)
    for i in range(2, len(lines), 4):
        for word in lines[i].split():
            words_list.append(word)
            increase_counter(words_dict, word)

    return words_list, words_dict


def increase_counter(words_counter_dict, word):
    words_counter_dict[word] = words_counter_dict.get(word, 0) + 1


def create_counter_dict(words):
    counter_dict = dict()
    for word in words:
        increase_counter(counter_dict, word)
    return counter_dict


def words_counter(filename):
    words_list, words_dict = get_words_list_and_dict(filename)
    split_index = round(0.9 * len(words_list))
    training_dict = create_counter_dict(words_list[:split_index])
    validation_dict = create_counter_dict(words_list[split_index:])

    return words_dict, training_dict, validation_dict


def init(outputs, words_dict, development_set_filename, test_set_filename, input_word,
         output_filename):
    outputs.append(development_set_filename)
    outputs.append(test_set_filename)
    outputs.append(input_word)
    outputs.append(output_filename)
    outputs.append(V_size)
    outputs.append(p(input_word, words_dict))


def development_set_preprocessing(outputs, words):
    outputs.append(sum(words.values()))


def lidstone_model_training(outputs, event, validation, training):
    outputs.append(sum(validation.values()))
    outputs.append(sum(training.values()))
    outputs.append(len(training.keys()))
    outputs.append(training.get(event, 0))
    outputs.append(p(event, training))
    outputs.append(p('unseen-word', training))
    outputs.append(p(event, training, 0.1))
    outputs.append(p('unseen-word', training, 0.1))


def write_outputs(outputs, output_filename):
    with open(output_filename, 'w') as file:
        file.write('#Students\tNoam Simon\tChen Eliyahou\t208388850\t312490675\n')
        for index in range(len(outputs)):
            file.write('#Output{0}\t{1}\n'.format(index + 1, outputs[index]))


def main(development_set_filename, test_set_filename, input_word, output_filename):
    outputs = list()
    words, training, validation = words_counter(development_set_filename)
    init(outputs, words, development_set_filename, test_set_filename, input_word, output_filename)
    development_set_preprocessing(outputs, words)
    lidstone_model_training(outputs, input_word, validation, training)
    write_outputs(outputs, output_filename)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
