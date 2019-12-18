class FrequencyTable:
    def __init__(self, held_out_model, lidstone_model, best_lambda, r_max):
        self.__frequency_table = dict()
        for r in range(r_max):
            self.__frequency_table[r] = (lidstone_model.get_frequency(r, best_lambda),
                                         held_out_model.get_frequency(r)) + held_out_model.get_prob_parameters(r)

    def to_string(self):
        table_string = '\n'
        for r, parameters in self.__frequency_table.items():
            table_string += '{}\t{:.5f} {:.5f} {:6d} {:6d}\n'.format(r, parameters[0], parameters[1], parameters[2],
                                                                       parameters[3])
        return table_string
