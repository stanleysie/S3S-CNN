import os
import cv2
import json
from tqdm import tqdm
from DataExtractor import DataExtractor
from OpticalFlow import OpticalFlow
from utils import read_file, write_data

def handle_windows_path(path):
    drive = path.split(':')[0]
    path = path.replace('\\', '/')
    return path.replace(f'{drive}:/', f'/mnt/{drive.lower()}/')

config = read_file('config.json')
config['data_dir'] = handle_windows_path(config['data_dir'])
config['facs_dir']['SAMM'] = handle_windows_path(config['facs_dir']['SAMM'])
config['facs_dir']['MMEW'] = handle_windows_path(config['facs_dir']['MMEW'])
config['facs_dir']['CASME_II'] = handle_windows_path(config['facs_dir']['CASME_II'])
config['out_dir'] = handle_windows_path(config['out_dir'])
config['raw_file'] = f"{config['out_dir']}/{config['raw_file']}"
config['dataset'] = f"{config['out_dir']}/{config['dataset']}"
config['face_bounding_box'] = f"{config['out_dir']}/{config['face_bounding_box']}"

extractor = DataExtractor(config['emotions'], 
                        config['data_dir'],
                        config['facs_dir'])
OpF = OpticalFlow(config['emotions'], config['db_OF'])

def get_onset_apex(emotions, dbs):
    print('\nExtracting onset and apex frames...')
    data = {}

    for emotion in tqdm(emotions):
        if not emotion in data.keys(): 
            data[emotion] = {
                'onset': [],
                'apex': []
            }
        
        for db in dbs:
            data[emotion] = extractor.get_data(db, emotion, data[emotion].copy())

    with open(f'{config["raw_file"]}', 'w') as f:
        json.dump(data, f, indent=4)


def generate_features(emotions, out_dir):
    print('\nGenerating features...')
    data = read_file(config['raw_file'])

    for expr in emotions:
        for i, (onset, apex) in enumerate(zip(data[expr]['onset'], data[expr]['apex'])):
            subject = onset[0] or apex[0]

            if 'SAMM' in onset[1] or 'SAMM' in apex[1]:
                db = 'SAMM'
            elif 'MMEW' in onset[1] or 'MMEW' in apex[1]:
                db = 'MMEW'
            elif 'CASME_II' in onset[1] or 'CASME_II' in apex[1]:
                db = 'CASME_II'

            print(f"{expr} - ({i+1}/{len(data[expr]['onset'])})", end=' | ')
            os.makedirs(f"{out_dir}/{db}", exist_ok=True)
            OpF.generate_optical_flow(i+1, f"{out_dir}/{db}", db, expr, subject, onset[1], apex[1])

    dataset = OpF.get_dataset()
    face_bounding_box = OpF.get_face_bounding_box()

    write_data(dataset, config['dataset'])
    write_data(face_bounding_box, config['face_bounding_box'])


def crop_resize(dim):
    print('\nCropping and resizing images...')
    dataset = read_file(config['dataset'])
    face_bounding_box = read_file(config['face_bounding_box'])

    for (OF, OS, H, V) in tqdm(dataset['X']):
        if 'SAMM' in OF or 'CASME_II' in OF:
            crop = True

            img = cv2.imread(OF)
            x, y, w, h = face_bounding_box[OF]
            OpF.crop_resize(OF, img, int(x), int(y), int(w), int(h), dim, crop)
            img = cv2.imread(OS)
            x, y, w, h = face_bounding_box[OS]
            OpF.crop_resize(OS, img, int(x), int(y), int(w), int(h), dim, crop)
            img = cv2.imread(H)
            x, y, w, h = face_bounding_box[H]
            OpF.crop_resize(H, img, int(x), int(y), int(w), int(h), dim, crop)
            img = cv2.imread(V)
            x, y, w, h = face_bounding_box[V]
            OpF.crop_resize(V, img, int(x), int(y), int(w), int(h), dim, crop)
        elif 'MMEW' in OF:
            crop = False

            img = cv2.imread(OF)
            OpF.crop_resize(OF, img, None, None, None, None, dim, crop)
            img = cv2.imread(OS)
            OpF.crop_resize(OS, img, None, None, None, None, dim, crop)
            img = cv2.imread(H)
            OpF.crop_resize(H, img, None, None, None, None, dim, crop)
            img = cv2.imread(V)
            OpF.crop_resize(V, img, None, None, None, None, dim, crop)


def main():
    get_onset_apex(config['emotions'], config['dbs'])
    generate_features(config['emotions'], config['out_dir'])
    crop_resize(config['img_dim'])

if __name__ == '__main__':
    main()
