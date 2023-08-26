import sys
# fix yolov7 imports
from pathlib import Path
sys.path.append(str(Path('yolov7').resolve()))

import torch
from torchvision import transforms
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint, plot_skeleton_kpts
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pickle
import gc
from tqdm import tqdm
import glob
import imageio
from pytube import YouTube


# if you have memory issies, try decreasing the batch size
BATCH_SIZE = 10


def download_video(video_url, save_path):
    # create videos folder if it doesn't exist
    if not os.path.exists('videos'):
        os.mkdir('videos')

    print('Downloading video...')
    try:
        yt = YouTube(video_url)
        stream = yt.streams.get_highest_resolution()  # You can choose different streams based on your preference
        stream.download(output_path='', filename=save_path)
        print("Download complete!")
    except Exception as e:
        print("An error occurred:", str(e))


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model


MODEL = load_model()


def load_image(url):
    image = cv2.imread(url) # shape: (480, 640, 3)
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (768, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 768, 960])
    return image


def run_inference(image):
    output, _ = MODEL(image.unsqueeze(0)) # torch.Size([1, large num, 57])
    return output


def run_batch_inference(images):
    output, _ = MODEL(torch.stack(images)) # torch.Size([N, large num, 57])
    return output


def get_fps(name):
    if '/' in name:
        # the inner fps
        return int(name.split('/')[-1].split('fps')[0])
    else:
        return int(name.split('fps')[0].split('_')[-1])


def extract_frames(path):
    ''' expects something like curry_full_court_5fps or curry_full_court_5fps/10_fps_split0 '''

    if '/' in path:
        # split_id is included in the name, such as curry_5fps/10fps_split_0
        name, split_str = path.split('/')
        fps = int(split_str.split('fps')[0])
    else:
        name = path
        fps = get_fps(name)

    print('extracting frames from video...')
    filenames = glob.glob(f'results/{name}/video.*')
    if len(filenames) > 1:
        raise ValueError(f'Found more than one file at {name}/video.*')
    video_path = filenames[0]

    folder = os.path.join('results', path, 'frames')

    # Create the output folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate the frame interval based on desired frames per second
    frame_interval = int(video_fps / fps)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frames at the desired interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print('done extracting')


def plot_keypoints_onto_axis(output, image, ax):
    output = non_max_suppression_kpt(output,
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=MODEL.yaml['nc'], # Number of Classes
                                     nkpt=MODEL.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    ax.imshow(nimg)


def plot_keypoints_in_grid(outputs, images):
    # Determine the number of images and the grid size
    num_images = len(outputs)
    grid_size = int(np.ceil(np.sqrt(num_images)))

    # Create a figure and set the grid layout
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    # Flatten the axes array if necessary
    if num_images == 1:
        axes = np.array([axes])

    # Iterate through the processed image list and display each image
    for i, (ax, output, image) in enumerate(zip(axes.flat, outputs, images)):
        if i < num_images:
            plot_keypoints_onto_axis(output, image, ax)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def get_keypoints_from_tensor(output):
    '''
    Input: a tensor of shape [N, X, 58], aka the one returned from calling
    run_inference on an image or images, and the same one that's passed into
    dot_output with the image to visualize the keypoints.

    Output: a list of keypoints for each image
      - the length of the list is N, the number of images
      - each index contains a list of keypoints for the corresponding image
      - each keypoint is an (x, y) coordinate
      - if no people are detected, the list is empty
      - if multiple people are detected, we use the person yolov7 was most confident was present
    '''
    output = non_max_suppression_kpt(output,
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=MODEL.yaml['nc'], # Number of Classes
                                     nkpt=MODEL.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        # [N, n_people, 58]
        outputs = [output_to_keypoint(output[i:i+1]) for i in range(len(output))]

    def get_keypoints_for_person(tensor):
        return [(int(tensor[7+ix*3]), int(tensor[7+ix*3+1])) for ix in range(17)]

    return [[] if len(output) == 0 else get_keypoints_for_person(output[0])
            for output in outputs]


def process_images_and_save_outputs(path, frames=None):
    print('processing images with yolov7 model...')
    if frames is None:
        frames = frame_filenames(path)

    images = [load_image(f) for f in frames]

    out_path = os.path.join('results', path, 'temp_results')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for ix, i in tqdm(enumerate(range(0, len(images), BATCH_SIZE))):
        output = run_batch_inference(images[i:i+BATCH_SIZE])
        with open(f'{out_path}/batch_{ix}.pkl', 'wb') as f:
            pickle.dump(output, f)
        del output
        gc.collect()

    print('done processing images')


def process_saved_outputs_to_keypoints(path, delete=True):
    # get all filenames of form '{path}/{name}_batch_{ix}.pkl'
    files = glob.glob(f'results/{path}/temp_results/batch_*.pkl')
    # sort by the number at the end
    files = sorted(files, key=lambda f: frame_num(f))

    all_keypoints = []
    for file in files:
        with open(file, 'rb') as f:
            output = pickle.load(f)
            keypoints = get_keypoints_from_tensor(output)
            all_keypoints += keypoints

    # save the keypoints list to a file
    with open(f'results/{path}/keypoints.pkl', 'wb') as f:
        pickle.dump(all_keypoints, f)

    if delete:
        for file in files:
            os.remove(file)
        os.rmdir(f'results/{path}/temp_results')


def load_keypoints(path):
    with open(f'results/{path}/keypoints.pkl', 'rb') as f:
        keypoints = pickle.load(f)
        return keypoints


def frame_filenames(path):
    folder_name = os.path.join('results', path, 'frames')
    filenames = os.listdir(folder_name)
    filenames = sorted(filenames, key=lambda f: frame_num(f))
    filenames = [os.path.join(folder_name, f) for f in filenames]
    return filenames


def keypoint_distance(keypoints1, keypoints2, translate=True):
    '''
    return the sum of the distances between the keypoints
    if either keypoints1 or keypoints2 is empty, returns None
    translate=True will translate the keypoints so that the first
    keypoint is at the origin, to attempt to be invariant to translation
    '''

    if not keypoints1 or not keypoints2:
        return None

    def dist(p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    if translate:
        # be invariant to initial starting location.
        # so get the location of one body part for each person, and subtract those coordinates from
        # all keypoints for each person
        x1, y1 = keypoints1[0]
        x2, y2 = keypoints2[0]

        keypoints1 = [(x - x1, y - y1) for (x, y) in keypoints1]
        keypoints2 = [(x - x2, y - y2) for (x, y) in keypoints2]

    return sum([dist(p1, p2) for p1, p2 in zip(keypoints1, keypoints2)])

def next_unused_path(path, extend_fn=lambda i: f'_{i}', return_i=False, start_zero=False):
    if '.' in path:
        last_dot = path.rindex('.')
        extension = path[last_dot:]
        file_name = path[:last_dot]
    else:
        extension = ''
        file_name = path

    if start_zero:
        path = file_name + extend_fn(0) + extension

    i = -1
    while os.path.exists(path):
        i += 1
        path = file_name + extend_fn(i) + extension

    if return_i:
        return path, i
    else:
        return path


def create_folder(video_path, fps=5):
    # create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.mkdir('results')

    name, ext = video_path.split('/')[-1].split('.')
    name = 'results/' + name + f'_{fps}fps'
    new_name = next_unused_path(name)
    if new_name != name:
        print(f'{name} already exists, saving instead at {new_name}')
        name = new_name

    os.mkdir(name)
    # copy the video into the folder
    os.system(f'cp {video_path} {name}/video.{ext}')
    # return name without results/ in front
    return name[8:]


def plot_img(image):
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(nimg)
    plt.show()


def frame_num(filename):
    return int(filename.split('_')[-1].split('.')[0])


def detect_splits(path, first_shot_ix, window_duration=1, n_shots=None, plot=False):
    '''
    window_duration - how long, in seconds, to use as the reference window
    min_shot_duration - how long, in seconds, to require until a new shot starts
    by default, min_shot_duration equals window_duration.

    returns the frame numbers where a new shot is predicted to start.
    '''

    filenames = frame_filenames(path)
    frame_nums = [frame_num(f) for f in filenames]

    keypoints = load_keypoints(path)

    fps = get_fps(path)
    window_size = window_duration * fps

    reference_window = keypoints[first_shot_ix:window_size+first_shot_ix]

    dists = []
    split_ixs = list(range(first_shot_ix + window_size, len(keypoints) - window_size))
    for ix in split_ixs:
        window = keypoints[ix:ix + window_size]
        window_dists = [keypoint_distance(reference_window[i], window[i]) for i in range(len(window))]
        window_dists = [d for d in window_dists if d is not None]
        avg_dist = -1 if not window_dists else np.mean(window_dists)
        dists.append(avg_dist)

    dists = [d if d != -1 else max(dists) for d in dists]
    min_ixs = local_minima(dists, n=None if n_shots is None else n_shots-1, plot=plot)
    # convert to actual split ixs - there's an offset from the window size
    predicted_split_ixs = [split_ixs[ix] for ix in min_ixs]
    # add back the ground truth start_ix
    predicted_split_ixs = [first_shot_ix] + predicted_split_ixs
    # convert to frame numbers
    predicted_splits = [frame_nums[ix] for ix in predicted_split_ixs]

    return predicted_splits


def local_minima(arr, n=None, plot=False):
    arr = np.array(arr)
    # smooth the array
    arr = np.convolve(arr, np.array([0.2, 0.6, 0.2]), mode='same')

    local_minima = [i for i in range(2, len(arr)-1) if arr[i-1] > arr[i] < arr[i+1]]
    local_minima = sorted(local_minima, key=lambda i: arr[i])
    local_maxima = [i for i in range(2, len(arr)-1) if arr[i-1] < arr[i] > arr[i+1]]

    # only include minima whose neighboring local maxima are at least 100 higher
    def steepness(i):
        befores = [m for m in local_maxima if m < i]
        afters = [m for m in local_maxima if m > i]
        prev_max = 0 if not befores else befores[-1]
        next_max = -1 if not afters else afters[0]
        return min(arr[prev_max] - arr[i], arr[next_max] - arr[i])

    # first minima whose steepness is less than 100 becomes the threshold, keep all minima before that
    steep_enoughs = [steepness(i) >= 90 for i in local_minima]
    pred_n = steep_enoughs.index(False) if False in steep_enoughs else len(local_minima)

    if n is None:
        minima = local_minima[:pred_n]
    else:
        minima = local_minima[:n]


    # for each false negative, print the steepness
    # if n is not None and pred_n < n:
        # for i in local_minima[pred_n:n]:
            # print(i, steepness(i))

    if plot:
        plt.plot(arr)
        # put a red dot at each predicted min (false pos)
        plt.scatter(local_minima[:pred_n], arr[local_minima[:pred_n]], c='r')
        if n is not None:
            # put a blue dot at each actual min (false neg)
            plt.scatter(local_minima[:n], arr[local_minima[:n]], c='b')
            # at predicted mins that are actual mins, put a purple dot (correct)
            plt.scatter(local_minima[:min(pred_n, n)], arr[local_minima[:min(pred_n, n)]], c='purple')
        plt.legend(['dist', 'false pos', 'false neg', 'correct'])
        plt.savefig(f'results/{path}/split_plot.png')
        # if n is not None:
            # plt.figure()
            # plt.scatter(range(n), [steepness(i) for i in local_minima[:n]], c='r')
            # plt.scatter(range(n, len(local_minima)), [steepness(i) for i in local_minima[n:]], c='k')
        plt.show()

    minima = sorted(minima)
    return minima


def plot_skeleton_keypoints_simon(im, keypoints):
    if len(keypoints) == 0:
        return

    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5

    for i, (x, y) in enumerate(keypoints):
        r, g, b = pose_kpt_color[i]
        if not (x % 640 == 0 or y % 640 == 0):
            cv2.circle(im, (int(x), int(y)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1, pos2 = keypoints[sk[0]-1], keypoints[sk[1]-1]
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def plot_keypoints(keypoints, image, save_path=None):
    nimg = image.permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    plot_skeleton_keypoints_simon(nimg, keypoints)

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(nimg)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close()


def save_annotated_frames(path):
    keypoints = load_keypoints(path)

    frames = frame_filenames(path)
    if '/' in path:
        frames = get_frames_in_windows(path)

    images = [load_image(f) for f in frames]

    if not os.path.exists(f'results/{path}/annotated_frames'):
        os.mkdir(f'results/{path}/annotated_frames')

    for i, (kpt, img) in enumerate(zip(keypoints, images)):
        plot_keypoints(kpt, img, save_path=f'results/{path}/annotated_frames/frame_{i}.jpg')


def create_mp4(path):
    input_path = f'results/{path}/annotated_frames/frame_%d.jpg'
    output_path = f'results/{path}/annotated.mp4'
    slowed = '' if '/' not in path else '-vf "setpts=4.0*PTS"'

    s = f'ffmpeg -framerate 30 -i {input_path} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {slowed} {output_path}'
    os.system(s)


def load_splits(path):
    with open(f'results/{path}/splits.txt', 'r') as f:
        return eval(f.readlines()[0])


def get_frames_in_windows(path):
    splits, window_size = load_splits(path)

    frames = frame_filenames(path)
    frame_nums = [frame_num(f) for f in frames]

    def ix_of_nearest_num(n, arr):
        return min(range(len(arr)), key=lambda i: abs(arr[i]-n))

    split_ixs = [ix_of_nearest_num(split, frame_nums) for split in splits]
    frames = [frames[ix:ix+window_size] for ix in split_ixs]
    # flatten list
    frames = [f for split in frames for f in split]
    return frames


def create_split_folder(path, splits, shot_duration, fps=10):
    # 1. extract frames/keypoints with new fps if needed
    num_frames = int(shot_duration * fps)

    path, i = next_unused_path(f'results/{path}/{fps}fps_split', return_i=True, start_zero=True)
    os.mkdir(path)
    path = path[8:] # remove 'results/' from the path

    with open(f'results/{path}/splits.txt', 'w') as f:
        f.write(str((splits, num_frames)))

    if i > 0:
        # just copy the frames from the other folder
        os.system(f'cp -r results/{path[:-2]}_0/frames results/{path}/frames')
    else:
        extract_frames(path)

    print('created split folder at', 'results/' + path)
    return path


def calculate_inconsistency(path):
    keypoints = load_keypoints(path)
    # if there are no keypoints, just copy the last frame's keypoints
    keypoints2 = []
    for k in keypoints:
        if len(k) == 0:
            keypoints2.append(keypoints2[-1])
        else:
            keypoints2.append(k)
    keypoints = keypoints2

    _, n_frames = load_splits(path)
    # [n_shots, n_frames, n_keypoints, 2]
    keypoints = np.array([keypoints[i:i+n_frames] for i in range(0, len(keypoints), n_frames)])
    n_shots, n_frames, n_keypoints, _ = keypoints.shape

    # translate based off the first keypoint in the first frame of each shot
    keypoints = keypoints - keypoints[:, 0:1, 0:1, :]

    # normalize by body size, so closer to camera doesn't mean more inconsistent
    # use the avg pairwise distance between keypoints as a proxy for body size
    differences = np.diff(keypoints[:, 0], axis=1) # [n_shots, n_keypoints-1, 2]
    assert differences.shape == (n_shots, n_keypoints-1, 2)
    body_sizes = np.linalg.norm(differences, axis=-1).mean(axis=-1)
    assert body_sizes.shape == (n_shots, )
    keypoints = keypoints / body_sizes[:, None, None, None]

    # std over shots, sum over x and y coordinates
    shot_std = np.std(keypoints, axis=0, ddof=1).sum(axis=-1)
    assert shot_std.shape == (n_frames, 17)
    inconsistency = 100 * shot_std.mean()
    print('inconsistency: ' + str(inconsistency))
    # save plain text to file 'inconsistency.txt'
    with open(f'results/{path}/inconsistency.txt', 'w') as f:
        f.write(str(inconsistency))

    return inconsistency


def process_splits(path, predicted_splits):
    split_path = create_split_folder(path, predicted_splits, shot_duration=2, fps=10)
    print(f'{split_path=}')

    frames = get_frames_in_windows(split_path)
    process_images_and_save_outputs(split_path, frames)
    process_saved_outputs_to_keypoints(split_path)
    save_annotated_frames(split_path)
    create_mp4(split_path)
    # delete the frames subdirectory
    os.system(f'rm -r results/{split_path}/frames')
    return split_path


if __name__ == '__main__':
    name = 'flips'
    download_video('https://www.youtube.com/watch?v=PfOcDtfZkO0', f'videos/{name}.mp4')
    path = create_folder(f'videos/{name}.mp4', fps=5)
    extract_frames(path)
    process_images_and_save_outputs(path)
    process_saved_outputs_to_keypoints(path)
    save_annotated_frames(path)
    create_mp4(path)
    first_shot_ix = int(input('input frame index where first shot starts:'))
    # n_shots = int(input('input number of shots:'))
    n_shots = None

    info = {
        'guy': ('guy_5fps', 9, 3),
        'curry': ('curry_full_court_5fps', 41, 5),
        'kid': ('kid_5fps', 11, 12),
        'pranay': ('pranay_5fps', 3, 5),
        'behind_short': ('behind_short_5fps', 6, 18),
        'bald': ('bald_5fps', 1, 11),
        'tennis': ('tennis_5fps', 0, 4),
        'trash': ('trash_5fps', 29, None),
        'flips': ('flips_5fps', 17, 5),
    }

    path, first_shot_ix, n_shots = info['flips']
    predicted_splits = detect_splits(path, first_shot_ix, window_duration=2,
                                     n_shots=n_shots, plot=True)

    split_path = process_splits(path, predicted_splits)

    split_path = f'{path}/10fps_split_0'

    inconsistency = calculate_inconsistency(split_path)
