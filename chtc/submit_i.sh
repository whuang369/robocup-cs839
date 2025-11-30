results_dir=results/interactive
log_dir=logs/interactive
mkdir -p ${results_dir}
mkdir -p ${results_dir}/logs
condor_submit -i job_configs/job_i.sub \
  results_dir=${results_dir} \
  log_dir=${log_dir}
