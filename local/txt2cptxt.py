import sys

org_txt = {}
with open(sys.argv[1]) as f:
    for l in f:
        l = l.split('\n')[0].split()
        if len(l) == 1:
            continue
        uttid, txt = l
        spkid, vad_index = uttid.split('-')
        if org_txt.get(spkid) is None:
            org_txt[spkid] = {}
        org_txt[spkid][int(vad_index)] = txt

with open(sys.argv[2], 'w') as f:
    contents = []
    for k, v in org_txt.items():
        vad_indexes = list(v.keys())
        vad_indexes.sort()
        contents.append(''.join([v[vad_index] for vad_index in vad_indexes]).replace('<unk>', ''))
    f.write('R8008_M8014 {}\n'.format('$'.join(contents)))