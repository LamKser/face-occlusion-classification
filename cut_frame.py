import os 
import cv2
import pandas as pd
import argparse
from tqdm import tqdm

def cut_frame(folder_video, label_csv, fps_step):

    if not os.path.exists('dataset/1'):
        os.makedirs('dataset/1')
    if not os.path.exists('dataset/0'):
        os.makedirs('dataset/0')

    print('Created 1 & 0 folder')

    df = pd.read_csv(label_csv)

    for name, label in tqdm(zip(df['fname'], df['liveness_score']),
                            total=len(df),
                            desc='Cutting frame'):
        video = cv2.VideoCapture(os.path.join(folder_video, name))
        fps = 0
        
        while video.isOpened():
            ret, frame = video.read()

            if ret:
                if fps % fps_step == 0:
                    cv2.imwrite(os.path.join('dataset', str(label), f'{name.split(".")[0]}_{fps}.jpg'), frame)
            else:
                break
            fps += 1
        video.release()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--folder-video', default='videos', type=str, metavar='--path',
                        help='Folder video path')
    parser.add_argument('--label-csv', default='label.csv', type=str, metavar='--csv',
                        help='Label file (.csv)')
    parser.add_argument('--fps-step', default=20, type=int, metavar='--fps',
                        help='Cutting frame after {fps}')

    args = parser.parse_args()
    cut_frame(args.folder_video, args.label_csv, args.fps_step)
    