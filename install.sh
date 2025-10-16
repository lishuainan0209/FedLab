#conda activate fedlab_P3
conda run -n fedlab_P3  pip uninstall fedlab -y
conda run -n fedlab_P3  pip install .
#echo "用户标识改为从1开始"