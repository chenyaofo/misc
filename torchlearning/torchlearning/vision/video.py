import cv2
import numpy

class VideoReadError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

class Video(object):
    def __init__(self, video_filename):
        self.video_filname = video_filename
        self.handler = cv2.VideoCapture(video_filename)
        if not self.handler.isOpened():
            raise VideoReadError(f"An unknown error is happened while opening '{video_filename}' with opencv.")
        self.n_frames = int(self.handler.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.handler.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.handler.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_channels = 3
        self.frames = []

    def get_all_frames(self):
        self.handler.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        while True:
            flag, frame = self.handler.read()
            if flag:
                self.frames.append(frame)
            else:
                return self.frames

    def frame_at(self, index: int):
        if index >= self.n_frames or index < 0:
            raise IndexError("Try to access a frame whose index({}) is out of the number({}) of video frames."
                             .format(index, self.n_frames))

        if self.frames:
            return self.frames[index]
        else:
            status = self.handler.set(cv2.CAP_PROP_POS_FRAMES, index)
            if status:
                flag, frame = self.handler.read()
                if flag:
                    return frame
                else:
                    return None
            else:
                raise Exception("Unkown error while accessing a frame from video({})."
                                .format(self.video_filname))

    def __del__(self):
        self.handler.release()