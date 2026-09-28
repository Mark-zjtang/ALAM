echo "***********************************************************"
if [ ! -d ${OUTPUT_ROOT} ]; then
   mkdir -p ${OUTPUT_ROOT}
fi

cd ${OUTPUT_ROOT}
dataset_path="${OUTPUT_ROOT}/task_ABC_D"
if [ -d ${dataset_path} ]; then
   	echo "${dataset_path} exists."
else
   echo "Downloading CALVIN task_ABC_D ..."
   wget http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
   unzip task_ABC_D.zip
   echo "saved folder: ${OUTPUT_ROOT}/task_ABC_D"
fi

echo "***********************************************************"
lmdb_path="${OUTPUT_ROOT}/lmdb_datasets/task_ABC_D/"
cd ${PROJECT_ROOT}/data_preprocessing
if [ -d ${lmdb_path} ]; then
   echo "${lmdb_path} exists."
else
   echo "Outputting CALVIN task_ABC_D lmdb dataset to ${lmdb_path} ..."
   python3 -u calvin_to_lmdb.py \
   --input_dir ${OUTPUT_ROOT}/task_ABC_D \
   --output_dir ${lmdb_path}
fi





<<COMMENT
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export OUTPUT_ROOT=[your path to save datasets]
cd ${PROJECT_ROOT}/scripts/
nohup bash download_and_preprocess_calvin_data.sh > download_and_preprocess_calvin_data.log 2>&1 &
tail -f download_and_preprocess_calvin_data.log
COMMENT