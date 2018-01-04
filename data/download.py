import os
import pandas as pd
import glob2 as glob

def read_data(dataset_path):

    new_types = ['badWeather', 'PTZ', 'turbulence', 'nightVideos', 'lowFramerate']    
    files = glob.glob(os.path.join(dataset_path, '**{}input{}*.*'.format(os.path.sep, os.path.sep)), recursive=True)
    files.sort()
    frame_list = []

    for filepath in files:
        info = filepath.split(os.path.sep)
        video_type = info[-4]
        video =  info[-3]
        input_frame = info[-1]

        frame_nb = int(info[-1].split('.')[0][2:])
        target_frame = 'gt{:06d}.jpg'.format(frame_nb)

        # videotype_dir = files[0][::-1].split(os.path.sep, maxsplit=2)[-1][::-1]
        videotype_dir = os.path.dirname(os.path.dirname(filepath))

        with open (os.path.join(videotype_dir,'temporalROI.txt'),'r') as f:
            roi_start, roi_end = [int(sample) for sample in f.readline().strip().split(' ')]

            if video_type in new_types:
                roi_end = int((roi_end + roi_start)/2) - 1

            if frame_nb < roi_start or frame_nb > roi_end:
                print('Ignoring {}'.format(filepath))
                continue

        frame_list.append([video_type, video, input_frame, target_frame])
    
    df = pd.DataFrame(frame_list, columns=['video_type', 'video_name', 'input_frame', 'target_frame'])  
    
    return df

def split(data, rate):
    if rate <= 0 or rate > 1:
        raise ValueError('rate must be in [0,1)')

    videos_split = [(video[:int(rate*len(video))], video[int(rate*len(video)):]) \
                        for _, video in df.groupby(['video_type', 'video_name'])]
        
    return zip(*video_split)

