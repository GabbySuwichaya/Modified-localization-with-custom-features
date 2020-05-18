colmap_path=$HOME"/colmap/build/src/exe/"

dataset="aachen-day-night" 
original_folder=$HOME"/Visuallocalizationbenchmark/local_feature_evaluation/data/"$dataset
initialized_database_file=$original_folder/database.db 

if [[ -d $original_folder"-Recon" ]]
then
    echo "[Warning] $original_folder-Recon already exists..."
else
    echo "[Warning] Creat $original_folder-Recon"
    mkdir $original_folder"-Recon"
fi
dataset_path_recon=$original_folder"-Recon"
mkdir $dataset_path_recon  

echo "################## Settings ############### "
  
ScoreThr=1.00 

model_file="models/d2_tf.pth" 
model_name=${model_file//"models/d2_"/""}
model_name=${model_name//".pth"/""}

method_name="d2_net"
FeatExtract="D2-Net" # "multi-local" 

echo $model_name 
 
suffix="$model_name"-off-the-shelf_Single-Scales_Scr"$ScoreThr"_Python_Origin_MuNNMatcher
 
suffix=${suffix//./p}
echo $suffix

d2_net_path_output="$dataset_path_recon/$method_name-$suffix"
SelectMETHOD=$method_name$suffix

mkdir $d2_net_path_output


echo "##################  Create folder ############### "

dataset_path=$d2_net_path_output
image_dir=$dataset_path/images_upright

if [[ -d $image_dir ]]
then
	echo "[Warning] $image_dir already exists..."
else
	echo "[Warning] $image_dir will be created..."
	ln -s $original_folder/images/images_upright $image_dir 
fi

image_dir=$image_dir/

match_filename=$dataset_path/match_list_exhaustivematching.txt
match_list=$dataset_path/$match_filename 


d2_net_path_output=$d2_net_path_output/feature

mkdir $d2_net_path_output 
  

echo "#########################  Extract features: extract_features_filter ######################### "


# Step 1. 
python3 genEmptyRecon_database.py  --dataset_path $dataset_path  --colmap_path $colmap_path 	--method_name $method_name

# Step 2.
python3 modify_database_with_custom_features_and_matches.py  --dataset_path $dataset_path/ --feat_path $d2_net_path_output --colmap_path $colmap_path  --method_name $method_name      --database_name $dataset_path/db.db  --image_path $image_dir  --match_list $dataset_path_recon/database_pairs_to_match.txt  
 
mkdir $dataset_path/model_$method_name

# Step 3.
colmap point_triangulator --database_path $dataset_path/$method_name.db --image_path $image_dir --input_path $dataset_path/sparse-d2_net-empty/ --output_path $dataset_path/model_$method_name/ --clear_points 1

# Step 4.
python modify_database_with_custom_features_and_matches.py --dataset_path $dataset_path/ --feat_path $d2_net_path_output --colmap_path $colmap_path  --method_name $method_name      --database_name $dataset_path/db.db  --image_path $image_dir  --match_list $dataset_path/query_to_database_pairs_to_match_20.txt  --matching_only True  

mkdir $dataset_path/"sparse-$method_name-final" 

# Step 5.
colmap image_registrator --database_path $dataset_path/$method_name.db --input_path $dataset_path/model_$method_name/ --output_path $dataset_path/"sparse-$method_name-final"

mkdir $dataset_path/"sparse-$method_name-final-txt"
# Step 6.
python3 extract_txt.py  --dataset_path $dataset_path  --colmap_path $colmap_path 	--method_name $method_name 
