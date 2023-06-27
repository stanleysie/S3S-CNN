import json

def read_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data


def write_json(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


def print_config(args):
    print(f'\n------------ Configurations -------------')
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('-------------- End ----------------\n')


def save_config(args, filename):
    with open(filename, 'w') as f:
        f.write(f'------------ Configurations -------------\n')
        for k, v in vars(args).items():
            f.write('%s: %s\n' % (str(k), str(v)))
        f.write('-------------- End ----------------\n')
    