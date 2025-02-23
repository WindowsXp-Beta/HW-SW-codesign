import itertools
import uuid
import subprocess
import os
import pandas as pd
import configparser
from pathlib import Path
from multiprocessing import Pool

config_dir = Path("system-configs-2b-1")
# Define parameter ranges
array_height_range = range(3, 11)
array_width_range = range(3, 11)
ifmap_sram_range = range(5, 11)
filter_sram_range = range(5, 11)
ofmap_sram_range = range(5, 11)
dataflow_options = ["ws", "os", "is"]
bandwidth_range = range(1, 6)

cycle_col_name = " Total Cycles"
map_eff_col_name = " Mapping Efficiency %"


def run_simulation(args):
    ah, aw, dfw = args
    results = {"names": [], "objective1": [], "objective2": []}

    for ifs, fs, ofs, bdw in itertools.product(
        ifmap_sram_range, filter_sram_range, ofmap_sram_range, bandwidth_range
    ):
        run_uuid = str(uuid.uuid4())  # Generate unique ID
        run_name = f"lenet_DSE_run_{run_uuid}"
        cfg_filename = f"system_{run_uuid}.cfg"

        # Read template configuration file
        template_config = configparser.ConfigParser()
        template_config.read("system.cfg")

        # Modify fields in the configuration
        template_config["general"]["run_name"] = run_name
        template_config["architecture_presets"]["ArrayHeight"] = str(ah)
        template_config["architecture_presets"]["ArrayWidth"] = str(aw)
        template_config["architecture_presets"]["IfmapSramSzkB"] = str(ifs)
        template_config["architecture_presets"]["FilterSramSzkB"] = str(fs)
        template_config["architecture_presets"]["OfmapSramSzkB"] = str(ofs)
        template_config["architecture_presets"]["Dataflow"] = dfw
        template_config["architecture_presets"]["Bandwidth"] = str(bdw)

        # Write updated configuration file
        with open(config_dir / cfg_filename, "w") as configfile:
            template_config.write(configfile)

        # Run scale-sim commands
        subprocess.run(
            [
                "/global/homes/h/hxu398/.conda/envs/pyro-slam/bin/python",
                "scale-sim-v2/scalesim/scale.py",
                "-c",
                str(config_dir / cfg_filename),
                "-t",
                "lenet_conv.csv",
                "-p",
                "DES_output_conv_2b_1",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "/global/homes/h/hxu398/.conda/envs/pyro-slam/bin/python",
                "scale-sim-v2/scalesim/scale.py",
                "-c",
                str(config_dir / cfg_filename),
                "-t",
                "lenet_gemm.csv",
                "-p",
                "DES_output_gemm_2b_1",
                "-i",
                "gemm",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Read results from CSVs
        conv_csv = f"DES_output_conv_2b_1/{run_name}/COMPUTE_REPORT.csv"
        gemm_csv = f"DES_output_gemm_2b_1/{run_name}/COMPUTE_REPORT.csv"

        if os.path.exists(conv_csv) and os.path.exists(gemm_csv):
            conv_df = pd.read_csv(conv_csv)
            gemm_df = pd.read_csv(gemm_csv)

            # Compute total cycles
            total_cycles = conv_df[cycle_col_name].sum() + gemm_df[cycle_col_name].sum()

            # Compute average mapping efficiency
            avg_mapping_efficiency = (
                conv_df[map_eff_col_name].sum()
                + gemm_df[map_eff_col_name].sum()
            ) / (len(conv_df) + len(gemm_df))

            # Compute total area
            total_area = 525 * (ah * aw) + 1015.7 * (ifs * 3 + fs * 3)

            # Compute objectives
            objective1 = 80000 / total_cycles + avg_mapping_efficiency * 8
            objective2 = 17500 / total_cycles + 15000 / total_area

            # Store results
            results["names"].append(cfg_filename)
            results["objective1"].append(objective1)
            results["objective2"].append(objective2)
        else:
            print(f"{run_name} missing report files!")

    return results


if __name__ == "__main__":
    with Pool() as pool:
        all_results = pool.map(
            run_simulation, itertools.product(array_height_range, array_width_range, dataflow_options)
        )

    # Merge results
    final_results = {"names": [], "objective1": [], "objective2": []}
    for res in all_results:
        final_results["names"].extend(res["names"])
        final_results["objective1"].extend(res["objective1"])
        final_results["objective2"].extend(res["objective2"])

    # Save results to CSV
    output_csv = "dse_results.csv"
    df_results = pd.DataFrame(final_results)
    df_results.to_csv(output_csv, index=False)

    print(f"DSE completed. Results saved in {output_csv}")
