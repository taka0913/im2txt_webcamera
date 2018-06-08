# im2txt on webcamera


## Description
Show and Tell realtime detection with webcamera based on "run_inference.py".


## Additional requirement
* **OpenCV**
    * to use cv2.VideoCapture()


## Installation
1.Clone this repository.

2.Copy "im2txt" folder and "process_camera.sh" to "~/models/research/im2txt".

3.Open "~/models/research/im2txt/im2txt/BUILD" and add this comment.
```
py_binary(
    name = "run_inference_camera",
    srcs = ["run_inference_camera.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":inference_wrapper",
        "//im2txt/inference_utils:caption_generator",
        "//im2txt/inference_utils:vocabulary",
    ],
)
```

4.Run this command at "~/models/research/im2txt".
```
bazel build -c opt //im2txt:run_inference_camera
```

## How to use
1.Open "process_camera.sh" and edit line to your path.
```
--checkpoint_path="${HOME}/im2txt/model/train" \
--vocab_file="${HOME}/im2txt/data/mscoco/word_counts.txt" \
```
2.Run this command at "~/models/research/im2txt".
```
chmod +x process_camera.sh
```
3.You can run.
```
./process_camera.sh
```

## Other
I can answer questions in English or Japanese.

日本語の質問でも大丈夫ですよ！
