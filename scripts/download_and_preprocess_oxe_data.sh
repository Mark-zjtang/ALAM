# DATASET_NAMES=('fractal20220817_data' 'kuka' 'bridge' 'taco_play' 'jaco_play' 'berkeley_cable_routing' 'roboturk' 'nyu_door_opening_surprising_effectiveness' 'viola' 'berkeley_autolab_ur5' 'toto' 'language_table' 'columbia_cairlab_pusht_real' 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds' 'nyu_rot_dataset_converted_externally_to_rlds' 'stanford_hydra_dataset_converted_externally_to_rlds' 'austin_buds_dataset_converted_externally_to_rlds' 'nyu_franka_play_dataset_converted_externally_to_rlds' 'maniskill_dataset_converted_externally_to_rlds' 'furniture_bench_dataset_converted_externally_to_rlds' 'cmu_franka_exploration_dataset_converted_externally_to_rlds' 'ucsd_kitchen_dataset_converted_externally_to_rlds' 'ucsd_pick_and_place_dataset_converted_externally_to_rlds' 'austin_sailor_dataset_converted_externally_to_rlds' 'austin_sirius_dataset_converted_externally_to_rlds' 'bc_z' 'usc_cloth_sim_converted_externally_to_rlds' 'utokyo_pr2_opening_fridge_converted_externally_to_rlds' 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds' 'utokyo_saytap_converted_externally_to_rlds' 'utokyo_xarm_pick_and_place_converted_externally_to_rlds' 'utokyo_xarm_bimanual_converted_externally_to_rlds' 'robo_net' 'berkeley_mvp_converted_externally_to_rlds' 'berkeley_rpt_converted_externally_to_rlds' 'kaist_nonprehensile_converted_externally_to_rlds' 'stanford_mask_vit_converted_externally_to_rlds' 'tokyo_u_lsmo_converted_externally_to_rlds' 'dlr_sara_pour_converted_externally_to_rlds' 'dlr_sara_grid_clamp_converted_externally_to_rlds' 'dlr_edan_shared_control_converted_externally_to_rlds' 'asu_table_top_converted_externally_to_rlds' 'stanford_robocook_converted_externally_to_rlds' 'eth_agent_affordances' 'imperialcollege_sawyer_wrist_cam' 'iamlab_cmu_pickup_insert_converted_externally_to_rlds' 'qut_dexterous_manipulation' 'uiuc_d3field' 'utaustin_mutex' 'berkeley_fanuc_manipulation' 'cmu_playing_with_food' 'cmu_play_fusion' 'cmu_stretch' 'berkeley_gnm_recon' 'berkeley_gnm_cory_hall' 'berkeley_gnm_sac_son' 'robot_vqa' 'droid' 'conq_hose_manipulation' 'dobbe' 'fmb' 'io_ai_tech' 'mimic_play' 'aloha_mobile' 'robo_set' 'tidybot' 'vima_converted_externally_to_rlds' 'spoc' 'plex_robosuite')
# DATASET_NAMES=('fractal20220817_data' 'bridge' 'taco_play' 'jaco_play' 'berkeley_cable_routing' 'roboturk' 'nyu_door_opening_surprising_effectiveness' 'viola' 'berkeley_autolab_ur5' 'toto')
DATASET_NAMES=('viola')
cd ${PROJECT_ROOT}/data_preprocessing

if [ ! -d "${OUTPUT_ROOT}/tensorflow_datasets/" ]; then
   mkdir -p "${OUTPUT_ROOT}/tensorflow_datasets/"
fi

for dataset_name in "${DATASET_NAMES[@]}"
do 
   dataset_path="${OUTPUT_ROOT}/tensorflow_datasets/${dataset_name}"
   if [ -d ${dataset_path} ]; then
   	echo "${dataset_path} exists."
   else
   	echo "Downloading ${dataset_name} to ${dataset_path} ..."
      gsutil -m cp -r gs://gresearch/robotics/${dataset_name} ${OUTPUT_ROOT}/tensorflow_datasets/
   fi

   video_path="${OUTPUT_ROOT}/tensorflow_datasets/${dataset_name}"
   if [ -d ${video_path} ]; then
   	echo "${video_path} exists."
   else
   	echo "Outputting ${dataset_name} videos to ${video_path} ..."
      python3 -u oxe_to_video.py \
      --dataset_name ${dataset_name} \
      --input_path ${OUTPUT_ROOT}/tensorflow_datasets/ \
      --output_path ${OUTPUT_ROOT}/video_datasets/
   fi

   lmdb_path="${OUTPUT_ROOT}/lmdb_datasets/${dataset_name}"
   if [ -d ${lmdb_path} ]; then
   	echo "${lmdb_path} exists."
   else
   	echo "Outputting ${dataset_name} lmdb dataset to ${lmdb_path} ..."
      python3 -u oxe_to_lmdb.py \
      --dataset_name ${dataset_name} \
      --input_dir ${OUTPUT_ROOT}/tensorflow_datasets/ \
      --output_dir ${OUTPUT_ROOT}/lmdb_datasets/
   fi
done

<<COMMENT
conda activate moto
pip install tensorflow-datasets
export PROJECT_ROOT=[your path to Moto project]
export OUTPUT_ROOT=[your path to save datasets]
cd ${PROJECT_ROOT}/scripts/
nohup bash download_and_preprocess_oxe_data.sh > download_and_preprocess_oxe_data.log 2>&1 &
tail -f download_and_preprocess_oxe_data.log
COMMENT