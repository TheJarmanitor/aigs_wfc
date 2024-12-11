
Tested with python 3.11

install dependencies with ```pip install -r requirements.txt```.
CPU version defined by default. If GPU is to be used, install with ```pip install -U "jax[cuda12]"```


How to run current pipeline:
1. Be in ...\aigs_wfc\ folder
2. command: 'py run.py'


<details>
 <summary>How to run some of the scripts</summary>

  - rule_split.py: 'py .\tools\rule_split.py [path_to_input_image] [tile_size] [output_folder]'
  - wfc.py: 'py .\tools\wfc.py [output_folder]'
  - visualize_wfc.py: 'py .\tools\visualize_wfc.py [input_folder] [input_txt_file] [output_image_file]'
</details>
