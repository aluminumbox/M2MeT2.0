# install necessary requirements, such as torch, torchaudio, speechbrain

# ts_vad decode
python3 local/ts_vad.py
md-eval.pl -acf -2 -c 0.25 -r data/fullref.rttm -s data/ts_vad.rttm

# ts_asr decode
python3 local/ts_asr.py

python3 local/txt2cptxt.py data/text data/text.cp
python3 local/txt2cptxt.py data/hyp data/hyp.cp
python3 local/compute_cpcer.py data/text.cp data/hyp.cp data/wer.cp