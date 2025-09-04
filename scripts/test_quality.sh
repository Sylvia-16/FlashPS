python scripts/cal_quality.py --dir1  "/flashps_data/standard_img_flux/" --dir2 "/flashps_data/flux_use_kv/" --dir3 "/flashps_data/flux_teacache/" --name1 "flashps" --name2 "teacache" --prompt_type "flux" --prompt_folder "/flashps_data/flux_prompts/" && \
echo "flux_flashps FID:" && \
python -m pytorch_fid  /flashps_data/standard_img_flux/collected_images /flashps_data/flux_use_kv/collected_images && \
echo "flux_teacache FID:" && \
python -m pytorch_fid  /flashps_data/standard_img_flux/collected_images  /flashps_data/flux_teacache/collected_images && \
python scripts/cal_quality.py --dir1 "/flashps_data/ootd_standard/" --dir2 "/flashps_data/ootd_use_o/" --dir3 "/flashps_data/ootd_teacache/" --name1 "flashps" --name2 "teacache" && \
echo "ootd_flashps FID:" && \
python -m pytorch_fid  /flashps_data/ootd_standard/  /flashps_data/ootd_use_o/ && \
echo "ootd_teacache FID:" && \
python -m pytorch_fid  /flashps_data/ootd_standard/  /flashps_data/ootd_teacache/ && \
python scripts/cal_quality.py --dir1 "/flashps_data/sd2_standard/" --dir2  "/flashps_data/sd2_use_kv/"  --dir3 "/flashps_data/sd2_fisedit/" --name1 "flashps" --name2 "fisedit" --prompt_type "sd2" --prompt_folder  "/flashps_data/sd2_prompts" && \
echo "sd2_flashps FID:" && \
python -m pytorch_fid  /flashps_data/sd2_standard/  /flashps_data/sd2_use_kv/ && \
echo "sd2_fisedit FID:" && \
python -m pytorch_fid  /flashps_data/sd2_standard/  /flashps_data/sd2_fisedit/