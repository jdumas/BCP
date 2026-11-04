import igl
import numpy as np
import pandas as pd

from compute_stats import compute_stats_from_V
from config_stats import get_cols_names_values_list
from config_stats import get_config
from func_utils import write_obj_with_UVs
from run_CM_utils import run_CM, run_CM_with_customized_init, run_CM_with_target_energy, \
    run_CM_with_target_runtime
from run_GIF_utils import run_GIF


def compute_stats_to_summary(
    results_df,
    source_path,
    tpath,
    pred_V,
    simplification_f_count,
    simplification_time,
    CM_time,
    mt_computation_time,
    CM_iters,
    simplification_Esd,
    time_GIF,
    IDT_stats,
    source_T,
    source_V,
    compare_to_GIF,
    compare_with_original_CM,
    compare_with_CM_with_our_initialization,
    compare_with_CM_until_ours_distortion,
    compare_with_CM_until_ours_runtime,
    stats_only,
    flipsData,
    root_dir_cache_CM,
    root_path_CM_exe,
    root_dir_cache_GIF,
    root_path_GIF_exe,
):
    ind = results_df.shape[0]

    config = get_config()
    cur_stats = {}
    cur_stats[config.stats_results_df_cols.MODEL_PATH] = source_path
    cur_stats[config.stats_results_df_cols.SIMPLIFICATION_E_SD] = simplification_Esd

    if IDT_stats is None:
        cur_stats[config.stats_results_df_cols.OURS_IDT_EDGE_FLIPS_COUNT] = 0
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_CONSUMPTION] = 0
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_ARE_ALL_COT_POSITIVE] = 0
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_MIN_COT_VALUE] = 0
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_NEGATIVE_COT_EDGES_COUNT] = 0
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_UNABLED_FLIPS] = 0
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_IS_SIMPLE_GRAPH] = 0
    else:
        cur_stats[config.stats_results_df_cols.OURS_IDT_EDGE_FLIPS_COUNT] = IDT_stats[0]
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_CONSUMPTION] = IDT_stats[8]
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_ARE_ALL_COT_POSITIVE] = IDT_stats[2]
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_MIN_COT_VALUE] = IDT_stats[3]
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_NEGATIVE_COT_EDGES_COUNT] = IDT_stats[4]
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_UNABLED_FLIPS] = IDT_stats[1]
        cur_stats[config.stats_results_df_cols.OURS_IDT_TIME_IS_SIMPLE_GRAPH] = IDT_stats[9]


    cur_stats[config.stats_results_df_cols.VERTICES_COUNT] = source_V.shape[0]
    cur_stats[config.stats_results_df_cols.FACES_COUNT] = source_T.shape[0]
    cur_stats[config.stats_results_df_cols.BOUNDARY_VERTICES_COUNT] = igl.boundary_loop(source_T).shape[0]
    cur_stats[config.stats_results_df_cols.SIMPLIFICATION_FACES_COUNT] = simplification_f_count


    cur_stats[config.stats_results_df_cols.OURS_SIMPLIFICATION_TIME] = simplification_time
    cur_stats[config.stats_results_df_cols.OURS_CM_SIMPLIFICATION_TIME] = CM_time
    cur_stats[config.stats_results_df_cols.OURS_CM_SIMPLIFICATION_ITERS] = CM_iters
    cur_stats[config.stats_results_df_cols.OURS_PROLONGATION_TIME] = mt_computation_time
    cur_stats[config.stats_results_df_cols.OURS_GIF_TIME] = time_GIF
    cur_stats[config.stats_results_df_cols.OURS_TOTAL_TIME_CONSUMPTION] = simplification_time + CM_time + mt_computation_time + time_GIF


    (
        cur_stats[config.stats_results_df_cols.OURS_E_SD],
        cur_stats[config.stats_results_df_cols.OURS_E_SD_BEST_99_99],
        cur_stats[config.stats_results_df_cols.OURS_E_SD_BEST_99_9],
        cur_stats[config.stats_results_df_cols.OURS_E_SD_BEST_99],
        cur_stats[config.stats_results_df_cols.OURS_E_SD_BEST_95],
        cur_stats[config.stats_results_df_cols.OURS_E_SD_WORST_5],
        cur_stats[config.stats_results_df_cols.OURS_E_SD_WORST_1],
        cur_stats[config.stats_results_df_cols.OURS_E_SD_MAX],
        cur_stats[config.stats_results_df_cols.OURS_K],
        cur_stats[config.stats_results_df_cols.OURS_AFTER_FINAL_FLIPS_POST_CHECK]
    ) = compute_stats_from_V(source_V, source_T, pred_V)


    cur_stats[config.stats_results_df_cols.OURS_BEFORE_MVC_FIX] = flipsData[3][0]
    cur_stats[config.stats_results_df_cols.OURS_AFTER_FINAL_FLIPS] = flipsData[4][0]
    cur_stats[config.stats_results_df_cols.OURS_MEAN_VALUE_FIX_WAS_APPLIED] = flipsData[8][0]

    if not stats_only:
        write_obj_with_UVs(
            tpath + "_source_textured_by_uv_ours.obj",
            source_V,
            source_T,
            pred_V,
        )

        igl.write_triangle_mesh(tpath + "_source_textured_by_uv_ours_uvs.obj", np.hstack([pred_V, np.zeros((pred_V.shape[0], 1))]), source_T)

    if compare_to_GIF:
        print("Comparing to GIF!")
        while True:
            i = 0
            try:
                res_GIF = run_GIF(source_V, source_T, root_path_GIF_exe, root_dir_cache_GIF, interior_faces=500)
                break
            except:
                i += 1
                print(f"{i}th attempt to run GIF failed")
                continue
        uvs_from_GIF, flipsDataGIF, time_consmption_GIF = (
            res_GIF[0],
            res_GIF[1],
            res_GIF[3],
        )

        cur_stats[config.stats_results_df_cols.GIF_BEFORE_MVC_FIX] = flipsDataGIF[3][0]
        cur_stats[config.stats_results_df_cols.GIF_AFTER_FINAL_FLIPS] = flipsDataGIF[4][0]
        cur_stats[config.stats_results_df_cols.GIF_MEAN_VALUE_FIX_WAS_APPLIED] = flipsDataGIF[8][0]

        cur_stats[config.stats_results_df_cols.GIF_TIME_CONSUMPTION] = time_consmption_GIF
        (
            cur_stats[config.stats_results_df_cols.GIF_E_SD],
            cur_stats[config.stats_results_df_cols.GIF_E_SD_BEST_99_99],
            cur_stats[config.stats_results_df_cols.GIF_E_SD_BEST_99_9],
            cur_stats[config.stats_results_df_cols.GIF_E_SD_BEST_99],
            cur_stats[config.stats_results_df_cols.GIF_E_SD_BEST_95],
            cur_stats[config.stats_results_df_cols.GIF_E_SD_WORST_5],
            cur_stats[config.stats_results_df_cols.GIF_E_SD_WORST_1],
            cur_stats[config.stats_results_df_cols.GIF_E_SD_MAX],
            cur_stats[config.stats_results_df_cols.GIF_K],
            _,
        ) = compute_stats_from_V(source_V, source_T, uvs_from_GIF)

        if not stats_only:
            write_obj_with_UVs(
                tpath + "_source_textured_by_uv_GIF.obj",
                source_V,
                source_T,
                uvs_from_GIF,
            )
            igl.write_triangle_mesh(tpath + "_source_textured_by_uv_GIF_uvs.obj", np.hstack([uvs_from_GIF, np.zeros((uvs_from_GIF.shape[0], 1))]), source_T)
    else:
        cur_stats[config.stats_results_df_cols.GIF_E_SD] = 0
        cur_stats[config.stats_results_df_cols.GIF_E_SD_BEST_99_99] = 0
        cur_stats[config.stats_results_df_cols.GIF_E_SD_BEST_99_9] = 0
        cur_stats[config.stats_results_df_cols.GIF_E_SD_BEST_99] = 0
        cur_stats[config.stats_results_df_cols.GIF_E_SD_BEST_95] = 0
        cur_stats[config.stats_results_df_cols.GIF_E_SD_WORST_5] = 0
        cur_stats[config.stats_results_df_cols.GIF_E_SD_WORST_1] = 0
        cur_stats[config.stats_results_df_cols.GIF_E_SD_MAX] = 0
        cur_stats[config.stats_results_df_cols.GIF_K] = 0
        cur_stats[config.stats_results_df_cols.GIF_BEFORE_MVC_FIX] = 0
        cur_stats[config.stats_results_df_cols.GIF_AFTER_FINAL_FLIPS] = 0
        cur_stats[config.stats_results_df_cols.GIF_MEAN_VALUE_FIX_WAS_APPLIED] = 0
        cur_stats[config.stats_results_df_cols.GIF_TIME_CONSUMPTION] = 0

    if compare_with_original_CM:
        uvs_from_CM, CM_time, CM_iters, _ = run_CM(source_V, source_T, root_path_CM_exe, root_dir_cache_CM)
        if np.isnan(uvs_from_CM).sum() > 0:
            raise Exception("nans in CM")
        cur_stats[config.stats_results_df_cols.CM_TIME] = CM_time
        cur_stats[config.stats_results_df_cols.CM_ITERS] = CM_iters
        (
            cur_stats[config.stats_results_df_cols.CM_E_SD],
            cur_stats[config.stats_results_df_cols.CM_E_SD_BEST_99_99],
            cur_stats[config.stats_results_df_cols.CM_E_SD_BEST_99_9],
            cur_stats[config.stats_results_df_cols.CM_E_SD_BEST_99],
            cur_stats[config.stats_results_df_cols.CM_E_SD_BEST_95],
            cur_stats[config.stats_results_df_cols.CM_E_SD_WORST_5],
            cur_stats[config.stats_results_df_cols.CM_E_SD_WORST_1],
            cur_stats[config.stats_results_df_cols.CM_E_SD_MAX],
            cur_stats[config.stats_results_df_cols.CM_K],
            cur_stats[config.stats_results_df_cols.CM_FLIPS],
        ) = compute_stats_from_V(source_V, source_T, uvs_from_CM)

        if not stats_only:
            write_obj_with_UVs(
                tpath + "_source_textured_by_uv_CM.obj",
                source_V,
                source_T,
                uvs_from_CM,
            )
            igl.write_triangle_mesh(tpath + "_source_textured_by_uv_CM_uvs.obj", np.hstack([uvs_from_CM, np.zeros((uvs_from_CM.shape[0], 1))]), source_T)
    else:
        cur_stats[config.stats_results_df_cols.CM_TIME] = 0
        cur_stats[config.stats_results_df_cols.CM_ITERS] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD_BEST_99_99] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD_BEST_99_9] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD_BEST_99] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD_BEST_95] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD_WORST_5] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD_WORST_1] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD_MAX] = 0
        cur_stats[config.stats_results_df_cols.CM_K] = 0
        cur_stats[config.stats_results_df_cols.CM_FLIPS] = 0

    if compare_with_CM_with_our_initialization:
        (
            _,
            CM_time_using_init,
            CM_iters_using_init,
            e_sd_using_init,
        ) = run_CM_with_customized_init(source_V, source_T, pred_V, root_path_CM_exe, root_dir_cache_CM)
        cur_stats[config.stats_results_df_cols.CM_TIME_WITH_OURS_INIT] = CM_time_using_init
        cur_stats[config.stats_results_df_cols.CM_ITERS_WITH_OURS_INIT] = CM_iters_using_init
        cur_stats[config.stats_results_df_cols.CM_E_SD_WITH_OURS_INIT] = e_sd_using_init
    else:
        cur_stats[config.stats_results_df_cols.CM_TIME_WITH_OURS_INIT] = 0
        cur_stats[config.stats_results_df_cols.CM_ITERS_WITH_OURS_INIT] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD_WITH_OURS_INIT] = 0

    if compare_with_CM_until_ours_distortion:
        (
            _,
            CM_time_target_energy,
            CM_iters_target_energy,
            e_sd_after_reaching_target_energy,
        ) = run_CM_with_target_energy(
            source_V,
            source_T,
            cur_stats[config.stats_results_df_cols.OURS_E_SD],
            root_path_CM_exe,
            root_dir_cache_CM,
        )
        cur_stats[config.stats_results_df_cols.CM_TIME_TO_REACH_OURS_ENERGY] = CM_time_target_energy
        cur_stats[config.stats_results_df_cols.CM_ITERS_TO_REACH_OURS_ENERGY] = CM_iters_target_energy
        cur_stats[config.stats_results_df_cols.CM_E_SD_AFTER_REACHING_OURS_ENERGY] = (
            e_sd_after_reaching_target_energy
        )
    else:
        cur_stats[config.stats_results_df_cols.CM_TIME_TO_REACH_OURS_ENERGY] = 0
        cur_stats[config.stats_results_df_cols.CM_ITERS_TO_REACH_OURS_ENERGY] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD_AFTER_REACHING_OURS_ENERGY] = 0

    if compare_with_CM_until_ours_runtime:
        (
            _,
            _,
            CM_iters_target_runtime,
            e_sd_after_reaching_target_runtime,
        ) = run_CM_with_target_runtime(
            source_V,
            source_T,
            cur_stats[config.stats_results_df_cols.OURS_TOTAL_TIME_CONSUMPTION],
            root_path_CM_exe,
            root_dir_cache_CM,
        )
        cur_stats[config.stats_results_df_cols.CM_ITERS_TO_REACH_OURS_RUNTIME] = CM_iters_target_runtime
        cur_stats[config.stats_results_df_cols.CM_E_SD_AFTER_REACHING_OURS_RUNTIME] = (
            e_sd_after_reaching_target_runtime
        )
    else:
        cur_stats[config.stats_results_df_cols.CM_ITERS_TO_REACH_OURS_RUNTIME] = 0
        cur_stats[config.stats_results_df_cols.CM_E_SD_AFTER_REACHING_OURS_RUNTIME] = 0

    if results_df.empty:
        config = get_config()
        df_cols = get_cols_names_values_list(config.stats_results_df_cols)
        results_df = pd.DataFrame(columns=df_cols)

    ordered_values_for_df = [cur_stats[col] for col in results_df.columns.values.tolist()]
    results_df.loc[ind] = ordered_values_for_df
    return results_df