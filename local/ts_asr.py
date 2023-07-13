import torch
import torchaudio
import yaml
import logging
from wenet.utils.init_model import init_model
from wenet.utils.checkpoint import load_checkpoint
import torchaudio.compliance.kaldi as kaldi
from wenet.utils.file_utils import read_symbol_table
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_fbank(waveform,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    waveform = waveform * (1 << 15)
    # Only keep key, feat, label
    mat = kaldi.fbank(waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    energy_floor=0.0,
                    sample_frequency=16000)
    return mat


if __name__=='__main__':
  data = torch.load('./data/R8008_M8014_bf.pt')
  signal, fs = torchaudio.backend.soundfile_backend.load('./data/R8008_M8014_bf.wav')
  personalized_speaker_embedding = data['personalized_speaker_embedding'].to(device)

  spk2rttm = {}
  with open('./data/ts_vad.rttm') as f:
    for l in f:
      l = l.split('\n')[0]
      _, _, _, start, dur, _, _, spk, _, _ = l.split()
      start, dur, spk = float(start), float(dur), int(spk)
      if spk2rttm.get(spk) is None:
        spk2rttm[spk] = []
      spk2rttm[spk].append({'start': start, 'dur': dur})

  with open('./data/train.yaml', 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
  model = init_model(configs)
  load_checkpoint(model, './data/ts_asr.pt')
  model = model.to(device)
  model.eval()

  symbol_table = read_symbol_table('data/units.txt')
  char_dict = {v: k for k, v in symbol_table.items()}
  eos = len(char_dict) - 1

  with torch.no_grad(), open('./data/hyp', 'w') as f:
    for k, v in tqdm(spk2rttm.items()):
      for i in tqdm(range(len(v))):
        if v[i]['dur'] < 0.1:
          continue
        this_signal = signal[:, int(v[i]['start'] * fs) : int((v[i]['start'] + v[i]['dur']) * fs)]
        feats = compute_fbank(this_signal).unsqueeze(dim=0).to(device)
        feats_lengths = torch.tensor([feats.shape[1]]).to(device)
        hyp, _ = model.attention_rescoring(
            feats,
            feats_lengths,
            personalized_speaker_embedding[:, :, k],
            10,
            decoding_chunk_size=-1,
            num_decoding_left_chunks=-1,
            ctc_weight=0.3,
            simulate_streaming=False,
            reverse_weight=0.3)
        content = []
        for w in hyp:
          if w == eos:
            break
          content.append(char_dict[w])
        f.write('R8008_M8014_{}-{} {}\n'.format(k, i, ''.join(content)))