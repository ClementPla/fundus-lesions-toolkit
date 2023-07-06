
labels2lesions = {
    0: 'BG',
    1: 'CTW',
    2: 'EX',
    3: 'HE',
    4: 'MA'
}

lesions2labels = {v:k for k,v in labels2lesions.items()}

lesions2names = {
    'BG': 'Background',
    'CTW': 'Cotton Wool Spots',
    'EX': 'Exudates',
    'HE': 'Hemorrhages',
    'MA': 'Microaneurysms',
}

names2lesions = {v:k for k,v in lesions2names.items()}

names2labels = {k:lesions2labels[v] for k,v in names2lesions.items()}

labels2names = {v:k for k,v in names2labels.items()}