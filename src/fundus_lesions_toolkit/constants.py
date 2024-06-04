from enum import Enum
LESIONS = ['BG', 'CTW', 'EX', 'HE', 'MA']
DEFAULT_COLORS = ['black', '#eca63f', '#8cf18e', '#4498f0', '#141488']

labels2lesions = {i:k for i,k in enumerate(LESIONS)}

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

lesions2colors = {l:c for l,c in zip(LESIONS, DEFAULT_COLORS)}

DEFAULT_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALIZATION_STD = (0.229, 0.224, 0.225)


class Dataset(str, Enum):
    IDRID: str = "IDRID"
    MESSIDOR: str = "MESSIDOR"
    DDR: str = "DDR"
    FGADR: str = "FGADR"
    RETINAL_LESIONS: str = "RETLES"
    RETLES: str = "RETLES"
    MAPLES: str = "MAPLES"
    MAPLES_DR: str = "MESSIDOR"
    ALL: str = "ALL"