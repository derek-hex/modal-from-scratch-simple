# cat masks and depth from a short video, using SAM 3 and Depth Anything 3 on Modal.
# setup notes: use Python 3.12; create a venv with python -m venv .venv.
# activate it: Windows .venv\Scripts\Activate.ps1; macOS/Linux source .venv/bin/activate.
# run python -m pip install -r requirements.txt, then python -m modal setup.
# Modal setup keeps your Modal token in ~/.modal.toml; do not put it in .env.
# request access to https://huggingface.co/facebook/sam3 and create a Hugging Face READ token.
# copy .env.example to .env beside this file and paste the token after HF_TOKEN=.
# save as UTF-8 without BOM; use an unquoted token with no spaces or substitutions.
# .env never goes into git; keep tokens out of screenshots and shared logs too.
# set planCredit below to your plan's monthly included credit in USD.
# run python cat_mask_depth_simple.py [video] [outFolder]; square brackets mean optional.
# no arguments uses the bundled sample; the default output folder is output.
# use a fresh output-something folder for other runs so .gitignore keeps results out of git.
# modal run and modal deploy are blocked: they skip the confirmation screen and .env check.
# type yes to launch; the estimate is not a spending cap. Check python -m modal app list after.
# outputs: masks/mask.0001.png and depth/depth.0001.exr sequences inside your output folder.
# SAM 3 and Depth Anything 3 install from their upstream repos under their own licenses.
# this version has not yet been run on Modal; the offline checks do not prove model compatibility.
# sample: "Kitten jumping on bamboo mat" by Commons user Fumikas Sagisavas, CC0 1.0.
# https://commons.wikimedia.org/wiki/File:Kitten_jumping_on_bamboo_mat.webm .

# tools used locally and on Modal
import io
import math
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
import modal

# settings you may change
appName = 'cat-mask-depth-simple'  # name shown on Modal
gpuType = 'L40S'  # GPU for both models
memoryMiB = 32768  # requested system memory, not GPU memory
cpuCores = 0.125  # requested cores for the GPU stage
timeout = 600  # GPU execution seconds
startupTimeout = 300  # startup seconds for either stage
fetchCpu = 2  # requested cores for weight fetching
fetchMemoryMiB = 4096  # requested memory for weight fetching
fetchTimeout = 600  # CPU weight-fetch execution seconds
prompt = 'cat'  # object to track
maxFrames = 16  # decode at most this many frames; allowed range is 1 to 130
maxInputBytes = 64 * 1024 * 1024  # local input limit: 64 MiB
maxPixels = 1920 * 1080  # pixel budget per decoded frame
expectedSeconds = 45  # measured 2026-11-11 on L40S with 16 frames

# rate = (1.95 L40S + 32 GiB x 0.008 + 0.125 cores x 0.0473) / 3600
gpuRatePerSecond = 0.000614  # Modal rates read 2026-11-11
planCredit = 30  # your plan's monthly credit in USD; Modal does not report it
samRepo = 'facebook/sam3'  # mask weights
depthRepo = 'depth-anything/DA3-BASE'  # depth weights
samCommit = 'f1647e8d0c1d280991f4d1b7d974c2704537fe66'  # SAM code revision
depthCommit = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'  # DA3 code revision
here = Path(__file__).resolve().parent  # folder beside this script
sampleVideo = here / 'sample' / 'kitten_jumping_on_bamboo_mat.webm'  # bundled sample
keysPath = here / '.env'  # credentials beside this script

# only python may launch locally; Modal's function containers import this file too
if modal.is_local() and __name__ != '__main__':
    raise SystemExit('Launch with: python cat_mask_depth_simple.py (not modal run or modal deploy)')

# describe the remote computer; these handles do not start anything
image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('git','libgl1','libglib2.0-0')
    .pip_install('torch==2.11.0','torchvision==0.26.0','xformers==0.0.35',
                 index_url='https://download.pytorch.org/whl/cu128')
    .pip_install(f'sam3 @ git+https://github.com/facebookresearch/sam3.git@{samCommit}',
                 f'depth-anything-3 @ git+https://github.com/'
                 f'ByteDance-Seed/Depth-Anything-3.git@{depthCommit}',
                 'numpy==1.26.4','opencv-python==4.11.0.86','pycocotools','addict',
                 'setuptools<81','huggingface_hub')
    .env({'HF_HOME': '/hf-cache','OPENCV_IO_ENABLE_OPENEXR': '1','PYTHONUNBUFFERED': '1'}))
weights = modal.Volume.from_name('hf-cache',create_if_missing=True)
app = modal.App(appName,image=image)

# download on cpu, then make the exact snapshots visible to the gpu
@app.function(volumes={'/hf-cache': weights},
    secrets=[modal.Secret.from_dotenv(keysPath.parent,filename=keysPath.as_posix())],
    cpu=fetchCpu,memory=fetchMemoryMiB,timeout=fetchTimeout,startup_timeout=startupTimeout,
    retries=0,max_containers=1,single_use_containers=True)
def fetchWeights():
    from huggingface_hub import snapshot_download

    # dependency error text could contain the token
    try:
        samSnapshot = snapshot_download(samRepo,allow_patterns=['sam3.pt','config.json'])
        depthSnapshot = snapshot_download(depthRepo)
    except Exception:
        message = 'Check HF_TOKEN and SAM 3 access; see the setup notes at the top of this file.'
        raise RuntimeError(message) from None
    weights.commit()
    return {samRepo: Path(samSnapshot).name,depthRepo: Path(depthSnapshot).name}

# helpers below run on the remote computer when processVideo calls them
# processVideo decodes the clip, makes masks, makes depth, then returns a zip
def runMasks(frames,count,plateSize,prompt,out,revision):
    import cv2
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_video_predictor
    checkpoint = hf_hub_download(samRepo,'sam3.pt',revision=revision)
    predictor = build_sam3_video_predictor(checkpoint_path=checkpoint)

    # disable output delays and young-track pruning for this short clip
    predictor.model.hotstart_delay = 0
    predictor.model.hotstart_unmatch_thresh = 0
    predictor.model.hotstart_dup_thresh = 0
    started = predictor.handle_request({'type': 'start_session','resource_path': str(frames)})
    session = started['session_id']
    first = predictor.handle_request({
        'type': 'add_prompt','session_id': session,'frame_index': 0,'text': prompt})
    perFrame = {int(first['frame_index']): first['outputs']}

    # follow the prompt forward; propagation may replace frame zero's result
    request = {
        'type': 'propagate_in_video',
        'session_id': session,
        'propagation_direction': 'forward',
        'start_frame_index': 0,
    }
    for item in predictor.handle_stream_request(request):
        perFrame[int(item['frame_index'])] = item['outputs']
    if set(perFrame) != set(range(count)):
        raise RuntimeError('SAM 3 did not return every frame.')

    # total each track's pixels and count its nonempty frames
    tracks = {}
    for item in perFrame.values():
        ids = np.asarray(item['out_obj_ids']).tolist()
        for index,track in enumerate(ids):
            area = int(np.asarray(item['out_binary_masks'][index]).sum())
            if track not in tracks:
                tracks[track] = (0,0)
            total,present = tracks[track]
            total += area
            if area > 0:
                present += 1
            tracks[track] = (total,present)

    # most pixels wins; ties use the number of nonempty frames
    chosen = None
    bestTotal = -1
    bestPresent = -1
    for track in tracks:
        total,present = tracks[track]
        if total > bestTotal or (total == bestTotal and present > bestPresent):
            chosen = track
            bestTotal = total
            bestPresent = present
    if bestTotal <= 0:
        print('gpu stage: nothing detected; writing black masks')

    # keep the chosen track across frames, with black masks when it is absent
    (out / 'masks').mkdir()
    for index,item in sorted(perFrame.items()):
        ids = np.asarray(item['out_obj_ids']).tolist()
        mask = np.zeros((plateSize[1],plateSize[0]),dtype=bool)
        if chosen in ids:
            mask = np.asarray(item['out_binary_masks'][ids.index(chosen)]).astype(bool)
        if mask.shape != (plateSize[1],plateSize[0]):
            raise RuntimeError('SAM 3 mask dimensions differ from the input.')

        # save black and white pixels as a numbered png
        maskPath = out / 'masks' / f'mask.{index + 1:04d}.png'
        maskPixels = np.where(mask,255,0).astype(np.uint8)
        if not cv2.imwrite(str(maskPath),maskPixels):
            raise RuntimeError('Could not write a mask.')

    # free the GPU for the depth model
    predictor.handle_request({'type': 'close_session','session_id': session})
    predictor.shutdown()
    del predictor,perFrame,first,item
    torch.cuda.empty_cache()

# ask DA3 for predictions, then write the depth files ourselves
def runDepth(paths,out,revision):
    import cv2
    import numpy as np
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(depthRepo,revision=revision)
    model = model.to('cuda')
    model = model.eval()
    framePaths = []
    for path in paths:
        framePaths.append(str(path))

    # omit export_dir and export_format; this script writes the depth files itself
    prediction = model.inference(framePaths,use_ray_pose=False,
        ref_view_strategy='saddle_balanced',process_res=504,process_res_method='upper_bound_resize')
    if len(prediction.depth) != len(paths) or not np.isfinite(prediction.depth).all():
        raise RuntimeError('Depth output count or finite-value check failed.')
    (out / 'depth').mkdir()

    # save unrounded float32 depth with zip compression inside each exr
    params = [
        cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_FLOAT,
        cv2.IMWRITE_EXR_COMPRESSION,cv2.IMWRITE_EXR_COMPRESSION_ZIP,
    ]
    for index,value in enumerate(prediction.depth):
        depthPath = out / 'depth' / f'depth.{index + 1:04d}.exr'
        depthPixels = np.asarray(value,dtype=np.float32)
        if not cv2.imwrite(str(depthPath),depthPixels,params):
            raise RuntimeError('Could not write an EXR.')

# decode, masks, depth, zip; always release the video and temporary files
@app.function(gpu=gpuType,volumes={'/hf-cache': weights},cpu=cpuCores,memory=memoryMiB,
    timeout=timeout,startup_timeout=startupTimeout,retries=0,
    max_containers=1,single_use_containers=True)
def processVideo(videoBytes,suffix,prompt,maxFrames,revisions):
    print('gpu stage: container started')

    # load only the weights that the cpu stage committed to the shared volume
    os.environ['HF_HUB_OFFLINE'] = '1'
    import cv2
    weights.reload()
    folder = tempfile.TemporaryDirectory()
    capture = None
    try:
        work = Path(folder.name)
        videoPath = work / ('input' + suffix)
        frames = work / 'frames'
        out = work / 'out'
        frames.mkdir()
        out.mkdir()
        videoPath.write_bytes(videoBytes)
        capture = cv2.VideoCapture(str(videoPath))
        paths = []

        # numeric filenames keep the decoded frames in order for SAM
        while len(paths) < maxFrames:
            ok,frame = capture.read()
            if not ok:
                break
            height,width = frame.shape[:2]
            if width * height > maxPixels:
                raise ValueError('Video exceeds the 1920 x 1080 pixel budget.')
            if paths and (width,height) != plateSize:
                raise ValueError('Video frame dimensions changed.')
            plateSize = (width,height)
            path = frames / f'{len(paths):05d}.png'
            if not cv2.imwrite(str(path),frame):
                raise RuntimeError('Could not write a decoded frame.')
            paths.append(path)
        capture.release()
        capture = None
        if not paths:
            raise ValueError('Video contains no decodable frames.')

        # finish masks and release SAM before loading the depth model
        print(f'gpu stage: decoded {len(paths)} frames')
        runMasks(frames,len(paths),plateSize,prompt,out,revisions[samRepo])
        print('gpu stage: masks done')
        runDepth(paths,out,revisions[depthRepo])
        print('gpu stage: depth done')

        # png and exr files are already compressed; store their relative paths
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(out.glob('*/*')):
                archive.write(path,path.relative_to(out).as_posix())
        return buffer.getvalue()
    finally:
        if capture is not None:
            capture.release()
        folder.cleanup()

# a missing credit adjustment means unknown, unless the plan has no credit
def readCredit():
    if planCredit == 0:
        return 0
    try:
        summary = modal.Workspace.from_context().billing.summary()
        for key,value in summary.adjustments.items():
            if key.replace('_',' ').strip().casefold() != 'credits':
                continue
            used = abs(float(value))
            if math.isfinite(used):
                return max(0,planCredit - used)
            return None
    except Exception:
        return None

# explain the estimate before asking to start anything on Modal
def confirmLaunch(videoName,frames,creditLeft):
    estimate = expectedSeconds * gpuRatePerSecond
    print(f'input: {videoName} (up to {frames} frames)')
    print(f'hardware: {gpuType} GPU + {memoryMiB / 1024:g} GiB memory')
    print(f'estimate: about ${estimate:.2f} '
          f'({expectedSeconds} s x ${gpuRatePerSecond:.6f}/s; rates read 2026-11-11)')
    if frames > 16:
        print('estimate: measured with 16 frames; more frames take longer')
    print('excludes: first image build, CPU weight step, storage and Modal\'s own retries')
    print('estimate: not a spending cap; actual time and usage vary')
    creditText = 'unknown: this run may be charged to your card'
    if creditLeft is not None:
        creditText = f'${creditLeft:.2f} of ${planCredit:.2f}'
    print(f'credit left: {creditText} (billing can lag)')
    if creditLeft is not None and creditLeft < estimate:
        print('credit: this run will be charged to your card; credit left is below the estimate')
    try:
        return input('Type yes to launch, anything else to cancel: ').strip().lower() == 'yes'
    except (EOFError,KeyboardInterrupt):
        return False

# check locally, confirm, run on Modal, then save into a fresh folder
def main():
    video = sampleVideo
    out = Path('output')

    # optional command-line values follow the script name
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    if len(sys.argv) > 2:
        out = Path(sys.argv[2])

    # catch local mistakes before reading billing or starting remote work
    videoTypes = ('.webm','.mp4','.mov','.mkv','.avi')
    if not video.is_file() or video.suffix.lower() not in videoTypes:
        raise ValueError('Input must be an existing video: webm, mp4, mov, mkv or avi.')
    if not 0 < video.stat().st_size <= maxInputBytes:
        raise ValueError('Input video must exist, be nonempty and at most 64 MiB.')
    if out.exists() or out.is_symlink():
        raise ValueError('Output folder already exists; choose a fresh folder.')
    if not out.parent.is_dir():
        raise ValueError('Output parent must be an existing directory.')
    if type(maxFrames) is not int or not 1 <= maxFrames <= 130:
        raise ValueError('Frame cap must be an integer from 1 to 130.')
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt) > 200:
        raise ValueError('Prompt must contain 1 to 200 characters.')

    # check credentials locally without showing their contents in errors
    try:
        if not keysPath.is_file():
            message = 'Missing .env. See the setup notes at the top of this file.'
            raise ValueError(message) from None
        contents = keysPath.read_text(encoding='utf-8',errors='strict')
    except (OSError,UnicodeError):
        message = 'Cannot read .env as UTF-8. See the setup notes at the top of this file.'
        raise ValueError(message) from None
    message = 'Invalid .env. See the setup notes at the top of this file.'
    if '\ufeff' in contents:
        raise ValueError(message) from None
    keys = set()

    # match dotenv's line endings so unicode separators cannot hide unchecked values
    for line in re.split('\r\n|\n|\r',contents):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key,mark,value = line.partition('=')
        if not mark or key not in ('HF_TOKEN','MODAL_MONTHLY_CREDIT_USD') or key in keys:
            raise ValueError(message) from None
        if key == 'HF_TOKEN' and not re.fullmatch('[A-Za-z0-9_-]+',value):
            raise ValueError(message) from None
        if key == 'MODAL_MONTHLY_CREDIT_USD' and not re.fullmatch(r'[0-9]+(\.[0-9]+)?',value):
            raise ValueError(message) from None
        keys.add(key)
    if 'HF_TOKEN' not in keys:
        raise ValueError(message) from None

    # read billing, then ask before starting the app
    creditLeft = readCredit()
    if not confirmLaunch(video.name,maxFrames,creditLeft):
        print('cancelled, nothing was started on Modal')
        return 0

    # check the bytes that will actually be uploaded
    videoBytes = video.read_bytes()
    if not 0 < len(videoBytes) <= maxInputBytes:
        raise ValueError('Input must be nonempty and at most 64 MiB.')

    # check that this destination accepts a file before paying for the run
    with tempfile.TemporaryFile(dir=out.parent):
        pass
    print('app: starting; the first run builds the image and can take minutes')
    archive = None  # the SDK can swallow Ctrl+C inside app.run
    try:
        with modal.enable_output():
            with app.run():
                print(f'app id: {app.app_id}')
                print('stage 1 of 2: checking model weights on a CPU')
                revisions = fetchWeights.remote()
                print(f'stage 2 of 2: asking Modal for an {gpuType} GPU '
                      '(this can take several minutes; nothing is wrong if it is quiet)')
                archive = processVideo.remote(videoBytes,video.suffix,prompt,maxFrames,revisions)
    except KeyboardInterrupt:
        archive = None
    finally:
        print('check stopped: python -m modal app list; python -m modal container list')
    if archive is None:
        print('interrupted: no outputs were delivered')
        return 1
    out.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive)) as saved:
        saved.extractall(out)
    print(f'outputs: {out.resolve().as_posix()}')
    return 0

# use python directly so confirmation happens before app.run
if __name__ == '__main__':
    raise SystemExit(main())
