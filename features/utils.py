import json

def read_file(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data


def write_data(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)