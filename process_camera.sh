# Run inference to generate realtime captions by webcamera.
bazel-bin/im2txt/run_inference_camera \
  --checkpoint_path="${HOME}/im2txt/model/train" \
  --vocab_file="${HOME}/im2txt/data/mscoco/word_counts.txt" \
