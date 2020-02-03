source activate py36
labelme
python labelme2mask.py --input_json $1
python main.py --filename_input $1 --filename_mask $1 --demo True
