import itertools
import uuid
import subprocess
import os
import pandas as pd
import configparser
from pathlib import Path
from multiprocessing import Pool

config_dir = Path("system-configs")
# Define parameter ranges
array_height_range = range(3, 11)
array_width_range = range(3, 11)
ifmap_sram_range = range(1, 5)
filter_sram_range = range(1, 5)
ofmap_sram_range = range(1, 5)
dataflow_options = ["ws", "os", "is"]

cycle_col_name = " Total Cycles"
map_eff_col_name = " Mapping Efficiency %"


def run_simulation(ah_aw):
    ah, aw = ah_aw
    results = {"names": [], "objective1": [], "objective2": []}

    for ifs, fs, ofs, df in itertools.product(
        ifmap_sram_range, filter_sram_range, ofmap_sram_range, dataflow_options
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
        template_config["architecture_presets"]["Dataflow"] = df

        # Write updated configuration file
        with open(config_dir / cfg_filename, "w") as configfile:
            template_config.write(configfile)

        # Run scale-sim commands
        subprocess.run(
            [
                "/home/hice1/xwei303/.conda/envs/hml/bin/python",
                "scale-sim-v2/scalesim/scale.py",
                "-c",
                str(config_dir / cfg_filename),
                "-t",
                "lenet_conv.csv",
                "-p",
                "DES_output_conv",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "/home/hice1/xwei303/.conda/envs/hml/bin/python",
                "scale-sim-v2/scalesim/scale.py",
                "-c",
                str(config_dir / cfg_filename),
                "-t",
                "lenet_gemm.csv",
                "-p",
                "DES_output_gemm",
                "-i",
                "gemm",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Read results from CSVs
        conv_csv = f"DES_output_conv/{run_name}/COMPUTE_REPORT.csv"
        gemm_csv = f"DES_output_gemm/{run_name}/COMPUTE_REPORT.csv"

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
            total_area = 525 * (ah * aw) + 1015.7 * (ifs + fs)

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
            run_simulation, itertools.product(array_height_range, array_width_range)
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
