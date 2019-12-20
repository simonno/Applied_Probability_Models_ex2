import math

from utils import create_counter_dict


class HeldOutModel:
    def __init__(self, dev_words_list, test_word_list, X, split_factor=0.5):
        if split_factor > 1 or split_factor < 0:
            raise Exception('Invalid split factor')
        self.__X = X
        self.__total_size = len(dev_words_list)
        self.__test_size = len(test_word_list)
        self.__training_size = round(self.__total_size * split_factor)
        self.__held_out_size = self.__total_size - self.__training_size
        self.__training = create_counter_dict(dev_words_list[:self.__training_size])
        self.__held_out = create_counter_dict(dev_words_list[self.__training_size:])
        self.__test = create_counter_dict(test_word_list)
        self.__N0 = X - len(self.__training.keys())
        self.__unseen_events = [event for event in self.__held_out.keys() if event not in self.__training.keys()]
        self.__create_prob_dict()

    def get_training_size(self):
        return self.__training_size

    def get_test_size(self):
        return self.__test_size

    def get_held_out_size(self):
        return self.__held_out_size

    def __create_prob_dict(self):

        self.__prob_dict = {0: (self.__calc_prob_parameters(0))}
        for r in self.__training.values():
            if r in self.__prob_dict.keys():
                continue
            self.__prob_dict[r] = (self.__calc_prob_parameters(r))

    def get_frequency(self, r):
        discounted_probability = self.__get_prob(r) if r in self.__prob_dict.keys() else 0
        return discounted_probability * self.__training_size

    def get_test_perplexity(self):
        sum_p = 0
        for event, count in self.__test.items():
            prob = self.get_prob(event)
            if prob == 0:
                sum_p = float(-math.inf)
                break
            else:
                sum_p += count * math.log(prob, 2)
        return math.pow(2, -sum_p / self.get_test_size())

    def get_prob(self, event):
        return self.__get_prob(self.__training.get(event, 0))

    def get_prob_parameters(self, r):
        return self.__prob_dict[r][1], self.__prob_dict[r][2]

    def __get_prob(self, r):
        return self.__prob_dict[r][0]

    def __calc_prob_parameters(self, r):
        if r == 0:
            r_events = self.__unseen_events
            Nr = self.__N0
        else:
            r_events = [event for event, count in self.__training.items() if count == r]
            Nr = len(r_events)

        tr = sum([count for event, count in self.__held_out.items() if event in r_events])

        return tr / (self.__held_out_size * Nr), Nr, tr

    def debug(self):
        sum_p = self.get_prob('unseen-word') * self.__N0
        for event in self.__training.keys():
            sum_p += self.get_prob(event)
        return sum_p
