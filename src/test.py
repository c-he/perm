from hair import load_hair, save_hair

strands = load_hair('strands00172.data')
print(strands.shape)
save_hair('strands00172-resaved.data', strands)

strands_reload = load_hair('strands00172-resaved.data')
print(strands_reload.shape)

diff = strands - strands_reload
print(f'diff: {diff.max()}')
