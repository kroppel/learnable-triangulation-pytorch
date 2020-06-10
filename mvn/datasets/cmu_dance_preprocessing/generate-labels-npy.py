#!/bin/python

'''
    Generate `labels.npy` for multiview `cmu-dance.py`
    Usage: `python3 generate-labels-npy.py <path/to/cmu-dance-data-root> <path/to/cmu-dance-bbox-detections>`
'''

import os, sys
import numpy as np
import json
import pickle

USAGE_PROMPT = """
$ python3 generate-lables-npy.py <path/to/data> <path/to/bbox-npy-file> <number-of-processors> <1-for-debug(optional)>

cmu-dance (default):
$ python3 generate-lables-npy.py $THIS_REPOSITORY/data/cmu $THIS_REPOSITORY/data/cmu-dance/cmu-dance-bboxes.npy 4
"""

# TODO: If your files are not in JSON format, need to change parser accordingly
def jsonToDict(filename):
    # Read file
    with open(filename, 'r') as f:
        data = f.read()

    # parse file
    return json.loads(data)


# TODO: Change this line if you want to use Mask-RCNN or SSD bounding boxes instead of H36M's 'ground truth'.
BBOXES_SOURCE = 'MRCNN' # or 'MRCNN' or 'SSD'

retval = {
    'cameras': None,
    'camera_names': set(),
    'action_names': [],
    'person_ids': set(),
    'table': []
}

try:
    cmu_dance_root = sys.argv[1]
    bbox_root = sys.argv[2]
    num_processes = int(sys.argv[3])

    USE_MULTIPROCESSING = False if num_processes <= 1 else True
except:
    print("Usage: ",USAGE_PROMPT)
    exit()

try:
    DEBUG = bool(sys.argv[4])
except:
    DEBUG = False

print(f"Debug mode: {DEBUG}\n")

destination_file_path = os.path.join(
    cmu_dance_root, f'cmu-dance-multiview-labels-{BBOXES_SOURCE}bboxes.npy')

assert os.path.isdir(cmu_dance_root), "Invalid data directory '%s'\n%s" % (cmu_dance_root, USAGE_PROMPT)

# Parse camera data and return a dictionary
# of better formatted data, with 
# key: camera name, value: dictionary of intrinsics
def parseCameraData(filename):
    # TODO: If your files are not in JSON format, need to change parser accordingly
    info_array = jsonToDict(filename)['cameras']
    data = {}

    for camera_params in info_array:
        # make it a number
        name = camera_params['name']

        data[name] = {}

        # TODO: Update according to JSON naming conventions
        data[name]['R'] = camera_params['R']
        data[name]['t'] = camera_params['t']
        data[name]['K'] = camera_params['K']
        data[name]['dist'] = camera_params['distCoef']

    return data

def parseBBOXData(bbox_dir):
    bboxes = np.load(bbox_dir, allow_pickle=True).item()

    return bboxes

# TODO: Modify according to BBOX Source
if BBOXES_SOURCE == 'MRCNN':
    print(f"Loading bbox data from {bbox_root}...")

    bbox_data = BBOXData(bbox_root)
    
    print(f"{BBOXES_SOURCE} bboxes loaded!\n")

    if DEBUG: 
        print(bbox_data)
else:
    # NOTE: If you are not using the provided MRCNN detections, you have to implement the parser yourself
    raise NotImplementedError

# In CMU data everything is by pose, not by person/subject
# NOTE: Calibration data for CMU is different for every pose, although only slightly :(
data_by_pose = {}

print(f"\nFinding actions, frames and cameras in {cmu_dance_root}")

def get_frames_data(images_dir_cam, camera_name):
    valid_frames = set()

    for frame_name in os.listdir(images_dir_cam):
        frame_name = frame_name.replace(f'{camera_name}_', '').replace(
            '.jpg', '').replace('.png', '')

        valid_frames.add(frame_name)

    return valid_frames

# TODO: Update according to data format/organisation
# Loop thru directory files and find scene names
for action_name in os.listdir(cmu_dance_root):
    # TODO: Update according to naming conventions (folder level)
    # Make sure that this is actually a scene
    # and not sth like 'scripts' or 'matlab'
    if '_dance' not in action_name and '_moonbaby' not in action_name:
        continue

    data = {}

    action_dir = os.path.join(cmu_dance_root, action_name)
    
    # Ensure is a proper directory
    if not os.path.isdir(action_dir):
        if DEBUG:
            print(f"{action_dir} does not exist")
        continue

    # Retrieve camera calibration data
    # TODO: Update according to naming conventions (file and folder level)
    calibration_file = os.path.join(action_dir, f'calibration_{action_name}.json')

    if not os.path.isfile(calibration_file):
        if DEBUG:
            print(f"{calibration_file} does not exist")
        continue
    
    camera_data = parseCameraData(calibration_file)

    # Count the frames by adding them to the dictionary
    # Only the frames with correct length are valid
    # Otherwise have missing data/images --> ignore
    frame_cnt = {}
    camera_names = []

    # Find the cameras
    images_dir = os.path.join(action_dir, 'hdImgs')

    if not os.path.isdir(images_dir):
        if DEBUG:
            print(f"Image directory {images_dir} does not exist")
        continue

    for camera_name in os.listdir(images_dir):
        # Populate frames dictionary
        images_dir_cam = os.path.join(images_dir, camera_name)

        for frame_name in os.listdir(images_dir_cam):
            frame_name = frame_name.replace(f'{camera_name}_', '').replace('.jpg', '')

            if frame_name in frame_cnt:
                frame_cnt[frame_name] += 1

        retval['camera_names'].add(camera_name)
        camera_names.append(camera_name)

    # Only add the action names when we know that this is valid
    retval['action_names'].append(action_name)
    data['action_dir'] = action_dir

    # Only frames with full count are counted
    valid_frames = []
    person_data = {}  # by frame name

    for frame_name in frame_cnt:
        if frame_cnt[frame_name] == 1 + len(camera_names):
            valid_frames.append(frame_name)

    del frame_cnt

    data['valid_frames'] = sorted(valid_frames)
    data['person_data'] = person_data
    data['camera_names'] = sorted(camera_names)

    # Generate camera data
    data['camera_data'] = {}
    for camera_name in data['camera_names']:
        data['camera_data'][camera_name] = camera_data[camera_name]

    data_by_pose[action_name] = data
    
# Consolidate camera names and sort them
retval['camera_names'] = list(retval['camera_names'])
retval['camera_names'].sort()

# Generate cameras based on len of names
# Note that camera calibrations are different for each pose
retval['cameras'] = np.empty(
    (len(retval['action_names']), len(retval['camera_names'])),
    dtype=[
        ('R', np.float32, (3, 3)),
        ('t', np.float32, (3, 1)),
        ('K', np.float32, (3, 3)),
        ('dist', np.float32, 5)
    ]
)

# TODO: Update according to how your data is organised
# TODO: and what data exists (or doesn't)
# TODO: for cmu-dance, you may not have ground truth keypoints

# Now that we have collated the data into easier-to-parse ways
# Need to reorganise data into return values needed for dataset class 

# Each pose, person has different entry
table_dtype = np.dtype([
    ('action_idx', np.int8), 
    ('person_id', np.int8),
    ('frame_name', np.int16),
    ('keypoints', np.float32, (19, 4)),   # TODO: Remove if no ground truth
    ('bbox_by_camera_tlbr', np.int16, (len(retval['camera_names']), 5))
])

if USE_MULTIPROCESSING:
    import multiprocessing
    print(f"\nGenerating labels using {num_processes} processors...")
else:
    print(f"\nGenerating labels. No multiprocessing used (see usage on how to setup multiprocessing)")

# print("NOTE: This may take a while (a few hours)!")
missing_data_file = "./missing-data-labels.txt"
missing_data = []

# TODO: Update according to data format/organisation
# Async process?
def load_table_segment(data, action_idx, action_name):
    person_ids = set()

    # NOTE: THIS IS AN ARRAY!
    table_segment = np.empty(len(data['valid_frames']), dtype=table_dtype)
    
    # Load BBOX: MRCNN Detections
    # let a (0,0,0,0,0) bbox mean that this view is missing
    # NOTE: Also an array, of size len(cameras)
    table_segment['bbox_by_camera_tlbr'] = 0

    for frame_idx, frame_name in enumerate(data['valid_frames']):
        person_data_arr = data['person_data'][frame_name]

        for person_data in person_data_arr:
            if DEBUG:
                print(
                    f"{action_name}, frame {frame_name}, person {person_data['id']}"
                )

            person_ids.add(person_data['id'])
            table_segment[frame_idx]['person_id'] = person_data['id']
            table_segment[frame_idx]['action_idx'] = action_idx
            table_segment[frame_idx]['frame_name'] = int(frame_name)

            # TODO: Remove if non-existent
            table_segment[frame_idx]['keypoints'] = person_data['joints']
            # Load BBOX: MRCNN Detections
            # let a (0,0,0,0,0) bbox mean that this view is missing
            for camera_idx, camera_name in enumerate(retval['camera_names']):
                try:
                    table_segment[frame_idx]['bbox_by_camera_tlbr'][camera_idx] = bbox_data[action_name][camera_name][int(frame_name)]
                except KeyError:
                    print(
                        f"Missing bbox data {action_name}, {camera_name}.. Ignoring")
                    missing_data.append((action_name, camera_name))
                    table_segment[frame_idx]['bbox_by_camera_tlbr'][camera_idx] = [0, 0, 0, 0, 0]

    return table_segment, person_ids

def save_segment_to_table(args):
    table_segment, person_ids = args

    retval['table'].append(table_segment)
    retval['person_ids'].union(person_ids)

if USE_MULTIPROCESSING:
    pool = multiprocessing.Pool(num_processes)
    async_errors = []

# Iterate through the poses to fill up the table and camera data
for action_idx, action_name in enumerate(retval['action_names']):
    data = data_by_pose[action_name]

    # For CMU, at most 31 cameras, so should be fast
    for camera_idx, camera_name in enumerate(data['camera_data']):
        if DEBUG: 
            # TODO: Update according to data organisation
            print(f"{action_name}, cam {camera_name}: ({action_idx},{camera_idx})")

        cam_retval = retval['cameras'][action_idx][camera_idx]
        camera_data = data['camera_data'][camera_name]

        cam_retval['R'] = np.array(camera_data['R'])
        cam_retval['K'] = np.array(camera_data['K'])
        cam_retval['t'] = np.array(camera_data['t'])
        cam_retval['dist'] = np.array(camera_data['dist'])

    if DEBUG:
        print("")

    # Load table segment
    if USE_MULTIPROCESSING:
        async_result = pool.apply_async(
            load_table_segment,
            args=(data, action_idx, action_name),
            callback=save_segment_to_table
        )

        async_errors.append(async_result)
    else:
        args = load_table_segment(data, action_idx, action_name)
        save_segment_to_table(args)

    if DEBUG: 
        print("\n")

if USE_MULTIPROCESSING:
    pool.close()
    pool.join()

    # raise any exceptions from pool's processes
    for async_result in async_errors:
        async_result.get()

retval['person_ids'] = sorted(list(retval['person_ids']))

# TODO: Remove the false flag if you want to check manually
# Check 
if DEBUG or False:
    print("\nChecking cameras:")
    for action_idx, action_name in enumerate(retval['action_names']):
        for camera_idx, camera_name in enumerate(retval['camera_names']):
            try:
                print(data_by_pose[action_name]["camera_data"][camera_name]['R'])
                print(retval['cameras'][action_idx][camera_idx]['R'])
                print("")
            except KeyError:
                print(f"Missing {action_name}, {camera_name}")

# Ready to Save!
retval['table'] = np.concatenate(retval['table'])
assert retval['table'].ndim == 1

print('Total frames in CMU Dance Dataset:', len(retval['table']))

print("\nSaving labels file...")

try:
    np.save(destination_file_path, retval)
    print(f"Labels file saved to {destination_file_path}")
except: 
    print(f"Failed to save file {destination_file_path}... Attempting to save in current directory instead...")

    try:
        np.save("./", retval)
        print(f"Labels file saved to current directory")
    except:
        raise Exception(f"Completely failed to save file: {destination_file_path}")

# Save list of missing bbox files
if DEBUG and len(missing_data) > 0:
    with open(missing_data_file, "w") as f:
        f.writelines("%s %s\n" % (action_name, camera_name)
                     for (action_name, camera_name) in missing_data)

    print(f"List of missing bbox data saved into {missing_data_file}")
