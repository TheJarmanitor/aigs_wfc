this is the readme

How to run current pipeline [EXAMPLE]:
1. Be in ...\wfc\aigs_wfc\ folder
2. run [powershell]: 'py .\tools\rule_split.py .\images\dragonwarr_island.png 16 dragon;  py .\tools\wfc.py dragon; py .\tools\visualize_wfc.py dragon'
    - or run them one by one:
        - 'py .\tools\rule_split.py .\images\dragonwarr_island.png 16 dragon'
        - 'py .\tools\wfc.py dragon'
        - 'py .\tools\visualize_wfc.py dragon'

3. The output will be in the 'outputs\dragon\output.png'

How to use scripts:
- rule_split.py: 'py .\tools\rule_split.py [path_to_input_image] [tile_size] [output_folder]'
- wfc.py: 'py .\tools\wfc.py [output_folder]'
- visualize_wfc.py: 'py .\tools\visualize_wfc.py [input_folder] [input_txt_file] [output_image_file]'