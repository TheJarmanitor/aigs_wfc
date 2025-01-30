
CPPN2WFC Pipeline for Map generation

Tested with python 3.11

install dependencies with ```pip install -r requirements.txt```.
CPU version defined by default. If GPU is to be used, install with ```pip install -U "jax[cuda12]"```


How to run Django App:
1. Be in ...\aigs_wfc\ folder
2. migration command: 'python manage.py migrate'
3. deplot app in local: 'python manage.py migrate'

The main page should send an error. The appropriate page needs to be run
There are three versions of the app, each with their respective page:

- Version A: The original pipeline. The page is /cppn-wfc/version=A/
- Version B: direct WFC encodings and Genetic Algorithms. The page is /cppn-wfc/version=B/
- Version C: Pure CPPN. The page is /cppn-wfc/version=C/


<details>
 <summary>How to run some of the scripts</summary>

  - rule_split.py: 'py .\tools\rule_split.py [path_to_input_image] [tile_size] [output_folder]'
  - wfc.py: 'py .\tools\wfc.py [output_folder]'
  - visualize_wfc.py: 'py .\tools\visualize_wfc.py [input_folder] [input_txt_file] [output_image_file]'
</details>
