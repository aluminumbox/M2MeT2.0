from tqdm import tqdm
import logging
import torch
from torch import nn
from resnet import Fbank, InputNormalization, ResNet34
from speechbrain.lobes.models.transformer.Conformer import ConformerDecoder
import torchaudio
import re


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 音频参数
fs = 16000
num_channel = 8
max_numspk = 4

# 训练参数
win_len = 1600
nframes = int(win_len * fs / 1e2)
seg_len = 300  # 返回frame-level embedding的seg_len与声纹训练时一致为3s
subsample = 8

# 解码参数
threshold = 0.5

# 模型参数
n_mels = 80
embed_dim = 256


class Decoder(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.decoder = ConformerDecoder(d_model=embed_dim, d_ffn=512, num_layers=2, nhead=int(512 / 64), attention_type='regularMHA')
    self.lstm = torch.nn.LSTM(input_size=embed_dim * max_numspk, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
    self.linear = torch.nn.Linear(512 * 2, max_numspk)

  def forward(self, embedding_frame, embedding_spk):
    # 送入encoder
    detection_state = []
    for i in range(embedding_spk.shape[2]):
      this_detection_state, _, _ = self.decoder(embedding_frame, embedding_spk[:, :, i: i + 1].transpose(1, 2))
      detection_state.append(this_detection_state)

    # 拼接detection state,送入bilstm
    detection_state = torch.concat(detection_state, dim=2)
    detection_state, _ = self.lstm(detection_state)

    # linear+sigmoid
    y_pred = torch.sigmoid(self.linear(detection_state))
    return y_pred


class Diarize(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.compute_features = Fbank()
    self.mean_var_norm = InputNormalization()
    self.embedding_model = ResNet34(feat_dim=n_mels, embed_dim=embed_dim, pooling_func='TSTP', two_emb_layer=False)
    self.gsp_mean = torch.nn.AvgPool1d(kernel_size=int(seg_len / subsample), stride=1, padding=int(seg_len / subsample / 2))
    self.attention_spk = nn.Sequential(
        nn.Conv1d(embed_dim, 64, kernel_size=1),
        nn.Tanh(),
        nn.Conv1d(64, embed_dim, kernel_size=1),
        nn.Softmax(dim=2),
    )
    self.linear = torch.nn.Linear(2560 * 3, 256)
    self.offline = Decoder()

  def frame_level_gsp(self, embedding):
    frame = embedding.shape[2]
    embedding_mean = self.gsp_mean(embedding)[:, :, :frame]
    embedding_std = torch.sqrt(self.gsp_mean((embedding - embedding_mean).pow(2)))[:, :, :frame]
    return torch.concat([embedding, embedding_mean, embedding_std], dim=1)

  def channel_level_sp(self, x):
    B, C, D, N = x.shape
    x = x.permute(0, 3, 2, 1).reshape(B * N, D, C)
    w = self.attention_spk(x)
    mu = torch.sum(x * w, dim=2)
    mu = x.mean(dim=2)
    x = mu.view(B, N, D).permute(0, 2, 1)
    return x

  def forward(self, x, embedding_spk):
    # 提取frame level embedding
    this_batchsize, this_num_channels = x.shape[0], x.shape[1]
    feat = self.compute_features(x.view(this_batchsize * this_num_channels, -1))
    feat = self.mean_var_norm(feat, torch.ones(len(feat)).to(x.device))
    embedding = self.embedding_model.forward_frame(feat, None)
    # frame level global_stats_pooling
    embedding = embedding.view(this_batchsize, this_num_channels, 2560, -1).mean(dim=1)
    embedding = self.frame_level_gsp(embedding).transpose(1, 2)
    embedding = torch.relu(self.linear(embedding))
    embedding_spk = self.channel_level_sp(embedding_spk)
    y_offline = self.offline(embedding, embedding_spk)
    return y_offline


class Interface():
  def __init__(self) -> None:
    self.model = Diarize().to(device)

  def eval(self, x, embedding_spk):
    y_offline = self.model(x, embedding_spk)
    return y_offline

  def load(self, fn):
    pth = torch.load(fn, map_location=device)
    self.model.load_state_dict({re.sub(r'^module.', '', k): v for k, v in pth['model'].items()})

def vad2rttm(vad):
  rttm = []
  speech, start = False, 0
  for i in range(vad.shape[0]):
    if vad[i] == 1 and speech == False:
      speech, start = True, i
    if vad[i] == 0:
      speech = False
    if (i == vad.shape[0] - 1 or vad[i + 1] == 0) and speech == True:
      rttm.append({'start': start, 'dur': i + 1 - start})
  assert vad.sum() == sum([i['dur'] for i in rttm])
  return rttm

if __name__ == '__main__':
  interface = Interface()
  interface.model.eval()
  for p in interface.model.parameters():
    p.requires_grad = False
  interface.load('./data/ts_vad.pt')

  logger.info('start eval')
  data = torch.load('./data/R8008_M8014.pt')

  signal, fs = torchaudio.backend.soundfile_backend.load('./data/R8008_M8014.wav')
  vad, personalized_speaker_embedding = data['vad'], data['personalized_speaker_embedding']
  this_numspk = personalized_speaker_embedding.shape[2]
  personalized_speaker_embedding = torch.concat([personalized_speaker_embedding, torch.zeros(8, 256, max_numspk - this_numspk)], dim=2)
  ts_vad = torch.zeros((vad.shape[0], max_numspk))

  indexes = [i for i in range(vad.shape[0]) if vad[i] != 0]
  start_len, end_len, hop_len = int(win_len * 0.25), int(win_len * 0.75), int(win_len * 0.5)
  for i in tqdm(range(0, len(indexes), hop_len)):
    this_indexes = indexes[i: i + win_len]
    if len(this_indexes) <= hop_len and len(this_indexes) != win_len:
      continue
    this_signal = torch.concat([signal[:, int(j * fs / 100): int((j + 1) * fs / 100)] for j in this_indexes], dim=1)
    this_y = interface.eval(this_signal.unsqueeze(dim=0).to(device), personalized_speaker_embedding.unsqueeze(dim=0).to(device)).squeeze(dim=0)
    this_start_len, this_end_len = start_len, end_len
    if this_indexes[0] == indexes[0]:
      this_start_len = 0
    if this_indexes[-1] == indexes[-1]:
      this_end_len = len(this_indexes)
    for j in range(this_numspk):
      for l in range(this_start_len, this_end_len):
        ts_vad[this_indexes[l], j] = this_y[min(int(l / subsample), this_y.shape[0] - 1), j]

  ts_vad = (ts_vad > threshold).float()
  with open('data/ts_vad.rttm', 'w') as f:
    for k in range(this_numspk):
      rttm = vad2rttm(ts_vad[:, k])
      for i in rttm:
        f.write('SPEAKER R8008_M8014 1 {} {} <NA> <NA> {} <NA> <NA>\n'.format(i['start'] / 100, i['dur'] / 100, k))