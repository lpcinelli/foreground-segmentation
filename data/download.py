import argparse
import hashlib
import os
import tarfile

import glob2 as glob
import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm

URL = "http://wordpress-jodoin.dmi.usherb.ca/static/dataset/dataset2014.zip"
MD5 = "d86332547edbc25f4ddbcd49f92413cf"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def md5file(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(url, md5sum, target_dir):
    """Download file from url to target_dir, and check md5sum.
    Adapted from PaddlePaddle/DeepSpeech repo"""
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    filepath = os.path.join(target_dir, url.split("/")[-1])
    if not (os.path.exists(filepath) and md5file(filepath) == md5sum):
        print("Downloading %s ..." % url)
        os.system("wget -c " + url + " -P " + target_dir)
        print("\nMD5 Chesksum %s ..." % filepath)
        if not md5file(filepath) == md5sum:
            raise RuntimeError("MD5 checksum failed.")
    else:
        print("File exists, skip downloading. (%s)" % filepath)
    return filepath


def unpack(filepath, target_dir, rm_tar=False):
    """Unpack the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    tar = tarfile.open(filepath)
    tar.extractall(target_dir)
    tar.close()
    if rm_tar == True:
        os.remove(filepath)


def read_data(data_dir, nb_bg_frames=150):

    new_types = [
        'badWeather', 'PTZ', 'turbulence', 'nightVideos', 'lowFramerate'
    ]
    print('Searching for images... ', end='')
    files = glob.glob(
        os.path.join(data_dir, '**{}input{}*.*'.format(os.path.sep,
                                                       os.path.sep)),
        recursive=True)
    files.sort()
    print(' Found {} files'.format(len(files)))
    frame_list = []
    bg_groups = {}

    print('Gathering information...')
    for filepath in tqdm(files):
        info = filepath.split(os.path.sep)
        video_type = info[-4]
        video_name = info[-3]
        input_frame = info[-1]

        frame_nb = int(info[-1].split('.')[0][2:])
        target_frame = 'gt{:06d}.png'.format(frame_nb)

        videotype_dir = os.path.dirname(os.path.dirname(filepath))

        with open(os.path.join(videotype_dir, 'temporalROI.txt'), 'r') as f:
            roi_start, roi_end = [
                int(sample) for sample in f.readline().strip().split(' ')
            ]

            img = np.asarray(Image.open(os.path.join(data_dir, video_type,
                                                video_name, 'groundtruth',
                                                target_frame)))
            negative_only = not img.any()

            # Ignoring half the frames in the new categories
            if video_type in new_types:
                roi_end = int((roi_end + roi_start) / 2) - 1

            if frame_nb <= nb_bg_frames:
                if '{}-{}'.format(video_type, video_name) not in bg_groups:
                    bg_groups['{}-{}'.format(video_type, video_name)] = []

                bg_groups['{}-{}'.format(video_type, video_name)].append(
                    np.asarray(
                        Image.open(
                            os.path.join(data_dir, video_type, video_name,
                                         'input', input_frame))))

            # Ignoring frames outside temporal roi
            if frame_nb < roi_start or frame_nb > roi_end:
                continue

            if video_type in new_types and \
                np.in1d(
                    np.unique(img),
                    np.array((85,170)),
                    assume_unique=True).all():
                # Ignoring frames w/o gt available though inside temporal roi
                    continue

        frame_list.append([
            video_type, video_name, input_frame, target_frame, negative_only])

    bg_groups = {
        k: np.median(np.stack(v, axis=0).astype(np.float), axis=0)
        for k, v in bg_groups.items()
    }

    df = pd.DataFrame(
        frame_list,
        columns=['video_type','video_name', 'input_frame',
            'target_frame', 'negative_only'])

    print('Saving bg models... ', end='')
    for k, v in bg_groups.items():
        video_type, video_name = k.split('-', maxsplit=1)

        Image.fromarray(v.astype(np.uint8)).save(
            os.path.join(data_dir, video_type, video_name, 'bg_model.jpg'))
    print('Ok!')

    print('Looking for corrupted ROI files... ', end='')
    for k, v in bg_groups.items():
        video_type, video_name = k.split('-', maxsplit=1)

        check_roi(os.path.join(data_dir, video_type, video_name))

    print('Ok!')

    return df


def check_roi(video_path):
    """Checks if ROI.bmp is readable, if it isn't uses ROI.jpg to create a new
    non-corrupted ROI.bmp
    """
    img_path = os.path.join(video_path, 'ROI.bmp')

    try:
        Image.open(img_path)
    except OSError:
        print("Corrupted '{}'. Creating a new one.".format(img_path))
        os.rename(img_path, img_path + '.bk')
        img = Image.open(os.path.join(video_path, 'ROI.jpg'))
        pix = np.array(img)
        mask = 255 * (pix.argmax(2) == 0).astype(pix.dtype)
        # mask = np.tile(mask[..., None], 3)
        Image.fromarray(mask).save(img_path)


def create_manifest(data_dir, manifest_path, rate):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. video type, video name, input frame, and
    target frame) of each video frame in the data set.
    """
    print("Creating manifests ...")
    data_df = read_data(data_dir)
    train, val = split(data_df, rate)

    train.to_csv('{}.train'.format(manifest_path), index=False)
    val.to_csv('{}.val'.format(manifest_path), index=False)


def prepare_dataset(url, md5sum, target_dir, manifest_path, rate):
    """Download, unpack and create summmary manifest file.
    """
    if not os.path.exists(os.path.join(target_dir, "dataset2014")):
        # download
        filepath = download(url, md5sum, target_dir)
        # unpack
        unpack(filepath, target_dir)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              target_dir)
    # create manifest csv file
    create_manifest(target_dir, manifest_path, rate)


def split(data, rate):
    if rate <= 0 or rate > 1:
        raise ValueError('rate must be in [0,1)')

    videos_split = [(video[:int(rate * len(video))],
                     video[int(rate * len(video)):])
                    for _, video in data.groupby(['video_type', 'video_name'])]

    train, val = zip(*videos_split)

    return pd.concat(
        train, ignore_index=True), pd.concat(
            val, ignore_index=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_dir",
        default=os.path.join(BASE_PATH, 'datafiles'),
        type=str,
        help="Directory to save the dataset. (default: %(default)s)")
    parser.add_argument(
        "--manifest_prefix",
        default="manifest",
        type=str,
        help="Filepath prefix for output manifests. (default: %(default)s)")
    parser.add_argument(
        "--rate",
        default=0.7,
        type=int,
        help="Train/val split ratio. (default %(default)s)")
    args = parser.parse_args()

    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(
        url=URL,
        md5sum=MD5,
        target_dir=args.target_dir,
        manifest_path=args.manifest_prefix,
        rate=args.rate)
