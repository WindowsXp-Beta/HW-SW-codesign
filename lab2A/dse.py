import itertools
import uuid
import csv
import subprocess
import os
import pandas as pd
import configparser
from multiprocessing import Pool

# Define parameter ranges
array_height_range = range(3, 11)
array_width_range = range(3, 11)
ifmap_sram_range = range(1, 5)
filter_sram_range = range(1, 5)
ofmap_sram_range = range(1, 5)
dataflow_options = ["ws", "os", "is"]


def run_simulation(ah_aw):
    ah, aw = ah_aw
    results = {"names": [], "objective1": [], "objective2": []}

    for ifs, fs, ofs, df in itertools.product(
        ifmap_sram_range, filter_sram_range, ofmap_sram_range, dataflow_options
    ):
        run_uuid = str(uuid.uuid4())[:8]  # Generate unique ID
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
        with open(cfg_filename, "w") as configfile:
            template_config.write(configfile)

        # Run scale-sim commands
        subprocess.run(
            [
                "python",
                "scale-sim-v2/scalesim/scale.py",
                "-c",
                cfg_filename,
                "-t",
                "lenet_conv.csv",
                "-p",
                "DES_output_conv",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "python",
                "scale-sim-v2/scalesim/scale.py",
                "-c",
                cfg_filename,
                "-t",
                "lenet_gemm.csv",
                "-p",
                "DES_output_gemm",
            ],
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
            total_cycles = conv_df["Total Cycles"].sum() + gemm_df["Total Cycles"].sum()

            # Compute average mapping efficiency
            avg_mapping_efficiency = (
                conv_df["Mapping Efficiency"].mean()
                + gemm_df["Mapping Efficiency"].mean()
            ) / 2

            # Compute total area
            total_area = 525 * (ah * aw) + 1015.7 * (ifs + fs)

            # Compute objectives
            objective1 = 80000 / total_cycles + avg_mapping_efficiency * 8
            objective2 = 17500 / total_cycles + 15000 / total_area

            # Store results
            results["names"].append(cfg_filename)
            results["objective1"].append(objective1)
            results["objective2"].append(objective2)

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
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["names", "objective1", "objective2"])
        for i in range(len(final_results["names"])):
            writer.writerow(
                [
                    final_results["names"][i],
                    final_results["objective1"][i],
                    final_results["objective2"][i],
                ]
            )

    print(f"DSE completed. Results saved in {output_csv}")
