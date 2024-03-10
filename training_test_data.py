from sklearn.model_selection import train_test_split

def ttd(tokens: list[str], window_size: int = 50):
    input_data = []
    label_data = []

    for i in range(len(tokens) - window_size - 1):
        input_data.append(tokens[i:i+window_size])
        label_data.append(tokens[i+1:i+window_size+1])

    return (input_data, label_data)
