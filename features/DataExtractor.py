import glob
import pandas as pd

class DataExtractor:
    def __init__(self, emotions, data_dir, facs_dir):
        self.emotions = emotions
        self.data_dir = data_dir
        self.facs_dir = facs_dir
        self.load_facs()
    

    def load_facs(self):
        samm_annotation = self.facs_dir['SAMM']
        samm_df = pd.read_csv(samm_annotation)
        samm_df.drop(labels=['Subject', 'Inducement Code',  'Micro', 'Objective Classes', 'Notes'], axis=1, inplace=True)
        samm_df.rename(columns={ 'Onset Frame': 'OnsetFrame', 'Apex Frame': 'ApexFrame', 'Offset Frame': 'OffsetFrame' }, inplace=True)
        samm_df['Subject'] = pd.Series([s.split('_')[0] for s in list(samm_df['Filename'])])
        samm_df = samm_df[samm_df['Estimated Emotion'].str.lower().isin(self.emotions)]
        samm_df = samm_df[['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame', 'Duration', 'Action Units', 'Estimated Emotion']]

        mmew_annotation = self.facs_dir['MMEW']
        mmew_df = pd.read_csv(mmew_annotation)
        mmew_df['Estimated Emotion'] = mmew_df['Estimated Emotion'].apply(lambda x: x.title())
        mmew_df['Duration'] = mmew_df['OffsetFrame'] - mmew_df['OnsetFrame'] + 1
        mmew_df.drop(labels=['remarks'], axis=1, inplace=True)
        mmew_df = mmew_df[mmew_df['Estimated Emotion'].str.lower().isin(self.emotions)]
        mmew_df = mmew_df[['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame', 'Duration', 'Action Units', 'Estimated Emotion']]

        casme_ii_annotation = self.facs_dir['CASME_II']
        casme_ii_df = pd.read_csv(casme_ii_annotation)
        casme_ii_df['Subject'] = casme_ii_df['Subject'].apply(lambda x: f'sub{x:02d}')
        casme_ii_df['Estimated Emotion'] = casme_ii_df['Estimated Emotion'].apply(lambda x: x.title())
        casme_ii_df['Duration'] = casme_ii_df['OffsetFrame'] - casme_ii_df['OnsetFrame'] + 1
        casme_ii_df = casme_ii_df[casme_ii_df['ApexFrame'] != '/']  # edge case
        casme_ii_df = casme_ii_df[casme_ii_df['Estimated Emotion'].str.lower().isin(self.emotions)]
        casme_ii_df = casme_ii_df[['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame', 'Duration', 'Action Units', 'Estimated Emotion']]

        self.samm_df = samm_df
        self.mmew_df = mmew_df
        self.casme_ii_df = casme_ii_df
    

    def get_data(self, db_name, emotion, data_dict):
        if db_name == 'SAMM':
            df = self.samm_df[self.samm_df['Estimated Emotion'].str.lower() == emotion]
            df_list = df.to_dict('records')

            image_path = f"{self.data_dir}/{db_name}/data"
            for d in df_list:
                if d['OnsetFrame'] == d['ApexFrame']:
                    continue
                onset_imgs = glob.glob(f"{image_path}/{d['Subject']}/{d['Filename']}/*{d['OnsetFrame']}.jpg")
                apex_imgs = glob.glob(f"{image_path}/{d['Subject']}/{d['Filename']}/*{d['ApexFrame']}.jpg")
                
                if len(onset_imgs) > 0:
                    data_dict['onset'].append((f"s_{d['Subject']}", onset_imgs[0]))
                if len(apex_imgs) > 0:
                    data_dict['apex'].append((f"s_{d['Subject']}", apex_imgs[0]))
        elif db_name == 'MMEW':
            df = self.mmew_df[self.mmew_df['Estimated Emotion'].str.lower() == emotion]
            df_list = df.to_dict('records')

            image_path = f"{self.data_dir}/{db_name}/data/{emotion}"
            for d in df_list:
                if d['OnsetFrame'] == d['ApexFrame']:
                    continue
                onset_imgs = glob.glob(f"{image_path}/{d['Filename']}/*{d['OnsetFrame']}.jpg")
                apex_imgs = glob.glob(f"{image_path}/{d['Filename']}/*{d['ApexFrame']}.jpg")
                
                if len(onset_imgs) > 0:
                    data_dict['onset'].append((f"m_{d['Subject']}", onset_imgs[0]))
                if len(apex_imgs) > 0:
                    data_dict['apex'].append((f"m_{d['Subject']}", apex_imgs[0]))
        elif db_name == 'CASME_II':
            df = self.casme_ii_df[self.casme_ii_df['Estimated Emotion'].str.lower() == emotion]
            df_list = df.to_dict('records')

            image_path = f"{self.data_dir}/{db_name}/data"
            for d in df_list:
                if d['OnsetFrame'] == d['ApexFrame']:
                    continue
                onset_imgs = glob.glob(f"{image_path}/{d['Subject']}/{d['Filename']}/img{d['OnsetFrame']}.jpg")
                apex_imgs = glob.glob(f"{image_path}/{d['Subject']}/{d['Filename']}/img{d['ApexFrame']}.jpg")
                
                if len(onset_imgs) > 0:
                    data_dict['onset'].append((f"c_{d['Subject']}", onset_imgs[0]))
                if len(apex_imgs) > 0:
                    data_dict['apex'].append((f"c_{d['Subject']}", apex_imgs[0]))

        data_dict['onset'] = list(set(data_dict['onset']))
        data_dict['apex'] = list(set(data_dict['apex']))
        data_dict['onset'] = sorted(data_dict['onset'], key=lambda x: x[1])
        data_dict['apex'] = sorted(data_dict['apex'], key=lambda x: x[1])
        
        return data_dict