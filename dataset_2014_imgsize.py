import pickle
with open('data/CROHME/crohme/2014/images.pkl','rb') as f:
    imgs = pickle.load(f)
sizes = {k: v.shape for k,v in imgs.items()}
largest = sorted(sizes.items(), key=lambda x: x[1][0]*x[1][1], reverse=True)[:5]
for name, shape in largest:
    print(f'{name}: {shape} = {shape[0]*shape[1]} px')