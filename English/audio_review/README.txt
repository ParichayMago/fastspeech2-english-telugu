Listen — local dataset review

From the assignment checkout:
  python3 English/audio_review/server.py
Open http://127.0.0.1:8766
Stop with Ctrl+C. A different port can be selected with --port.
No dependencies, GPU, cloud calls or external assets are required.

Choose Final assignment audio to review the ten required recordings.
Use Dataset recordings for source-data review.

Use Language, Dataset split and Speaker to choose recordings. Search transcripts
or sample IDs. Play, scrub the waveform, change playback speed, and mark Good,
Needs review or Reject. Add issue tags and listening notes. A/B holds two clips,
which helps compare voice consistency within an estimated speaker cluster.

Keys when focus is outside form controls: Space play/pause; arrows previous/next;
1 Good, 2 Needs review, 3 Reject. Buttons and all review controls also work by mouse.

Reviews are written atomically to ignored data/audio-review/ratings.json. The save
indicator reports failures; click it to retry. Do not reload if unsaved work is
reported. Existing recordings, feature arrays and train/validation splits are
never modified. Good selections do not automatically become new training data.
English speaker labels remain estimates; flags are automatic review prompts,
not a determination that audio is poor. Selected durations count original source
audio, not aligned feature hours.

Export good selections downloads a CSV of the current source/language/split/
speaker scope, with source paths, sample hashes, transcript, duration, review
notes and original split. Search, status and flagged filters narrow the listening
queue; they do not change the export scope. Review files and exports can be used
later to prepare a separately validated training selection.

Catalog records are fixed for a server session. Restart after generating new
samples or changing manifests. Reviews for files whose audio hash changed are
not reused. The server binds only 127.0.0.1 and serves allowlisted audio IDs and
three static UI files. It never serves the repository as a directory.

Verification:
  python3 -m unittest discover -s English/audio_review -p 'test_*.py' -v
