from distutils.core import setup
setup(
  name = 'fundus_lesions_toolkit',         # How you named your package folder (MyLib)
  packages = ['fundus_lesions_toolkit'],   # Chose the same as "name"
  version = '0.1.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Tools Lesions Segmentation in Fundus',   # Give a short description about your library
  author = 'Clement Playout',                   # Type in your name
  author_email = 'clement.playout@polymtl.ca',      # Type in your E-Mail
  url = 'https://github.com/ClementPla/fundus-lesions-toolkit/tree/main',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/ClementPla/fundus-lesions-toolkit/archive/refs/tags/V_011.tar.gz',    # I explain this later on
  keywords = ['Segmentation', 'Fundus', 'Pytorch', 'Models', 'Semantic', 'Lesions'],   # Keywords that define your package best
  install_requires=[            
          'torch',
          'nntools',
          'torchvision',
          'opencv-python',
          'numpy',
          'matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License', 
    'Programming Language :: Python :: 3',]
)