from numpy import array

def _tokens_to_learning_data(sample, max_seq_len):
    n = len(sample)
    if n <= max_seq_len:
        return ([], [])

    input_data = []
    label_data = []

    for i in range(n - (max_seq_len+1)):
        input_data.append(sample[i:i+max_seq_len])
        label_data.append(sample[i+max_seq_len])

    return input_data, label_data



def get_learning_data(samples, max_seq_len):
    input_data = []
    label_data = []

    for sample in samples:
        new_input_data, new_label_data = _tokens_to_learning_data(sample, max_seq_len)

        input_data += new_input_data
        label_data += new_label_data

    return array(input_data), array(label_data)