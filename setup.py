from distutils.core import setup
setup(
  name = 'fundus_lesions_toolkit',        
  packages = ['fundus_lesions_toolkit'],  
  version = '0.1.0',
  license = 'MIT', 
  description = 'Tools Lesions Segmentation in Fundus', 
  author = 'Clement Playout',                   
  author_email = 'clement.playout@polymtl.ca', 
  url = 'https://github.com/ClementPla/fundus-lesions-toolkit/tree/main',   
  download_url = 'https://github.com/ClementPla/fundus-lesions-toolkit/archive/refs/tags/v0.1.0.tar.gz',   
  keywords = ['Segmentation', 'Fundus', 
              'CNN', 'Neural Networks',
              'PyTorch', 'Models', 
              'Semantic', 'Lesions'],  
  install_requires = ['torch', 'nntools', 'torchvision', 'opencv-python', 'numpy', 'matplotlib'],
  classifiers = [
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License', 
    'Programming Language :: Python :: 3',]
)