import os
import cv2
from torchlearning.vision import Video
from torchlearning.utils import chunks
from multiprocessing import Process

VIDEO_FORMATS = ["mp4", "avi", "mkv", "webm", "flv", "3gp", "mov", "wmv", "ts", "m3u8", "mpg", "mpeg", "rm", "ram"]


def is_video(name):
    basename = os.path.basename(name)
    _, suffix = os.path.splitext(basename)
    suffix = suffix[1:]
    return suffix.lower() in VIDEO_FORMATS


def parse_video(source, depth=2):
    video_list = []
    if depth == 1:
        for video in os.listdir(source):
            if is_video(video):
                video_list.append((source, video))
        return video_list
    elif depth == 2:
        for category in os.listdir(source):
            category_path = os.path.join(source, category)
            if os.path.isdir(category_path):
                for video in os.listdir(category_path):
                    if is_video(video):
                        video_list.append((source, category, video))
        return video_list


def extract_frames_one_video(source_tuple, destination):
    video = Video(os.path.join(*source_tuple))
    frames = video.get_all_frames()
    _, *remians = source_tuple
    destination_path = os.path.join(destination, *remians)
    os.makedirs(destination_path, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(destination_path, "{:05d}.jpg".format(i)), frame)

def extract_frames_wrap(source_tuple_list,destination):
    for i,source_tuple in enumerate(source_tuple_list):
        print("{}/{},{}".format(i,len(source_tuple_list),os.path.join(*source_tuple)))
        extract_frames_one_video(source_tuple,destination)

def extract_frames(source, destination, depth=2, n_process=8):
    video_list = parse_video(source, depth)
    video_list_split = chunks(video_list,n_process)
    processes = []
    for i in range(n_process):
        processes.append(Process(target=extract_frames_wrap,args=(video_list_split[i],destination,)))

    for p in processes:
        p.start()
