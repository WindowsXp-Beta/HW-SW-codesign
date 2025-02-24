import itertools
import uuid
import subprocess
import os
import pandas as pd
from pathlib import Path
import configparser

config_dir = Path("system-configs-2b")
# Define parameter ranges
array_height_range = range(3, 11)
array_width_range = range(3, 11)
dataflow_options = ["ws", "os", "is"]

# Fixed values for SRAM sizes
ifs = 5
fs = 5
ofs = 5

cycle_col_name = " Total Cycles"
map_eff_col_name = " Mapping Efficiency %"


def run_simulation():
    results = {"names": [], "avg_map_eff": []}

    for ah, aw, df in itertools.product(
        array_height_range, array_width_range, dataflow_options
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
                "DES_output_conv_2b",
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
                "DES_output_gemm_2b",
                "-i",
                "gemm",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Read results from CSVs
        conv_csv = f"DES_output_conv_2b/{run_name}/COMPUTE_REPORT.csv"
        gemm_csv = f"DES_output_gemm_2b/{run_name}/COMPUTE_REPORT.csv"

        if os.path.exists(conv_csv) and os.path.exists(gemm_csv):
            conv_df = pd.read_csv(conv_csv)
            gemm_df = pd.read_csv(gemm_csv)

            # Compute total cycles

            # Compute average mapping efficiency
            avg_mapping_efficiency = (
                conv_df[map_eff_col_name].sum() + gemm_df[map_eff_col_name].sum()
            ) / (len(conv_df) + len(gemm_df))

            # Compute total area
            # total_area = 525 * (ah * aw) + 1015.7 * (ifs + fs)

            # Compute objectives
            # objective1 = 80000 / total_cycles + avg_mapping_efficiency * 8
            # objective2 = 17500 / total_cycles + 15000 / total_area

            # Store results
            results["names"].append(cfg_filename)
            results["avg_map_eff"].append(avg_mapping_efficiency)
        else:
            print(f"{run_name} missing report files!")

    return results


if __name__ == "__main__":
    final_results = run_simulation()

    # Save results to CSV using pandas
    output_csv = "dse_results_2b.csv"
    df_results = pd.DataFrame(final_results)
    df_results.to_csv(output_csv, index=False)

    print(f"DSE completed. Results saved in {output_csv}")
