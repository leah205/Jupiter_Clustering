import ast

import config.dicts as D
import config.types as T
import scripts.pipeline.pipeline as PIPE
from scripts.data_formatting.parse_map_to_df import parse_map_to_df


def run_all_cluster_ROI(r: dict):
    """
        Runs clustering pipeline on region of interest for each clustering run specified in config

        Parameters
        ----------------
        r: row in pd data frame


        Returns
        ------------
    void
    """

    ROI = ast.literal_eval(r["ROI Dict"].split("=")[1])
    flag = 0
    source = r["Data Source"]

    # if (not "GRS" in r["Name"]) and (not "Oval" in r["Name"]):
    #     return

    if "20251016" in r["Name"]:
        return

    if "test" in r["Name"]:
        return
    # if r["Name"] != "20251016UTc-SEZ-SED-Ext":
    #     return

    lat_lims, lon_lims = r["lat_lims"], r["lon_lims"]

    # mapConfig = T.mappingConfig(
    #     keywords=["NH3", "PCld"],
    #     ROI=ROI,
    #     latRng=lat_lims,
    #     lngRng=lon_lims,
    #     name=r["Name"],
    #     source=source,
    #     cm_num=int(r["Sys"]),
    # )

    # clusterConfig = T.clusterConfig(n_comp=4, isPca=False)
    # config = T.pipelineConfig(map=mapConfig, cluster=clusterConfig)

    # for i in range(8):
    #     PIPE.run_full_pipeline(config, i)
    for cl_info in D.cluster_runs:
        mapConfig = T.mappingConfig(
            keywords=cl_info["dims"],
            ROI=ROI,
            latRng=lat_lims,
            lngRng=lon_lims,
            name=r["Name"],
            source=source,
            cm_num=int(r["Sys"]),
        )

        for n_comp in cl_info["comps"]:
            clusterConfig = T.clusterConfig(n_comp=n_comp, isPca=cl_info["PCA"])
            config = T.pipelineConfig(map=mapConfig, cluster=clusterConfig)
            PIPE.run_full_pipeline(config)


if __name__ == "__main__":
    df = parse_map_to_df()
    # run_all_cluster_ROI(df.iloc[0])

    # print(df.head())
    df.apply(run_all_cluster_ROI, axis=1)
