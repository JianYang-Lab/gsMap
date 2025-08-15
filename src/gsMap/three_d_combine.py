import scanpy as sc
import pandas as pd
import logging
import os
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.multitest as smm
import pyvista as pv

from pathlib import Path
from scipy.stats import fisher_exact
from gsMap3D.three_d_plot.three_d_plots import three_d_plot, three_d_plot_save
from gsMap3D.cauchy_combination_test import acat_test
from gsMap3D.config import ThreeDCombineConfig
from pandas.core.frame import DataFrame

pv.start_xvfb()
logger = logging.getLogger(__name__)


def combine_ldsc(args):

    # Set the output path
    ldsc_root = Path(args.project_dir) / "3D_combine" / "spatial_ldsc"
    ldsc_root.mkdir(parents=True, exist_ok=True)
    name = ldsc_root / f"{args.trait_name}.csv.gz"
    
    # Merge all the ldsc results
    pth = Path(args.project_dir) / "spatial_ldsc"
    sldsc_pth = []
    for slice in os.listdir(pth):
        filtemp = pth / slice / f"{slice}_{args.trait_name}.csv.gz"
        if filtemp.exists():
            sldsc_pth.append(filtemp)

    if not os.path.exists(name):
        logger.info(f"Find {len(sldsc_pth)} ST sections for {args.trait_name}, start to merge the results...")
        # Load the results
        ldsc_merge = pd.DataFrame()
        for idx, file in enumerate(sldsc_pth):
            ldsc_temp = pd.read_csv(file, compression="gzip")
            ldsc_temp["ST_id"] = file.name.split(f"_{args.trait_name}")[0]
            # print(ldsc_temp.head())
            ldsc_merge = pd.concat([ldsc_merge, ldsc_temp], axis=0)
            
        # Check the cell name duplication
        if (ldsc_merge.spot.value_counts() > 1).any():
            logger.info('There are duplicated spot names, using the st_id + spot_id as the spot index.')
            ldsc_merge['spot_index'] = ldsc_merge['ST_id'] + '_' + ldsc_merge['spot'].astype(str)
        else:
            ldsc_merge['spot_index'] = ldsc_merge['spot']
            
        # save the merged results
        ldsc_merge.to_csv(name, compression="gzip", index=False)
        logger.info(f"Saving the 3D merged results to {name}")
    else:
        logger.info(f"The merged gsMap results already exist, loading the merged results from {name}...")
        ldsc_merge = pd.read_csv(name, compression="gzip")
    
    return ldsc_merge


def cauchy_combination_3d(ldsc):
    p_cauchy = []
    p_median = []
    gc_median = []
    for ct in np.unique(ldsc.annotation):
        p_temp = ldsc.loc[ldsc["annotation"] == ct, "p"]
        z_temp = ldsc.loc[ldsc["annotation"] == ct, "z"]
        p_temp = p_temp.dropna()
        
        # The Cauchy test is sensitive to very small p-values, so extreme outliers should be considered for removal...
        p_temp_log = -np.log10(p_temp)
        median_log = np.median(p_temp_log)
        IQR_log = np.percentile(p_temp_log, 75) - np.percentile(p_temp_log, 25)

        p_use = p_temp[p_temp_log < median_log + 3 * IQR_log]
        z_use = z_temp[p_temp_log < median_log + 3 * IQR_log]
        n_remove = len(p_temp) - len(p_use)

        # Outlier: -log10(p) < median + 3IQR && len(outlier set) < 20
        # if 0 < n_remove < max(len(p_temp) * 0.001,100):
        if 0 < n_remove < len(p_temp) * 0.05:
            print(
                f"Remove {
                    n_remove}/{len(p_temp)} outliers (median + 3*IQR) for {ct}."
            )
            p_cauchy_temp = acat_test(p_use)
        else:
            p_cauchy_temp = acat_test(p_temp)

        p_median_temp = np.median(p_use)
        gc_median_temp = np.median(z_use**2) / 0.4549
        
        p_cauchy.append(p_cauchy_temp)
        p_median.append(p_median_temp)
        gc_median.append(gc_median_temp)

    data = {
        "p_cauchy": p_cauchy,
        "p_median": p_median,
        "inflation_factor": gc_median,
        "annotation": np.unique(ldsc.annotation),
    }
    p_tissue = pd.DataFrame(data)
    p_tissue.columns = ["p_cauchy", "p_median", "inflation_factor", "annotation"]
    p_tissue.sort_values("p_cauchy", inplace=True)
    return p_tissue

# def cauchy_combination_3d(args):
    
#     # Load the cauchy combination results of each ST slices
#     pth = Path(args.project_dir) / "cauchy_combination"
#     st_file = os.listdir(pth)
#     logger.info(f"Find {len(st_file)} sections of cauchy combination results for {args.trait_name}...")
    
#     cauchy_all = pd.DataFrame()
#     for slice in st_file:
#         filtemp = pth / slice / f"{slice}_{args.trait_name}.Cauchy.csv.gz"
#         if filtemp.exists():
#             cauchy = pd.read_csv(filtemp, compression="gzip")
#             cauchy_all = pd.concat([cauchy_all, cauchy], axis=0)
            
#     cauchy_all = cauchy_all[~cauchy_all.annotation.isna()]
    
#     # Do the cauchy combination test across slices
#     p_cauchy = []
#     p_median = []
#     for ct in cauchy_all.annotation.unique():
#         cauchy_temp = cauchy_all.loc[cauchy_all.annotation == ct]
#         p_cauchy_temp = cauchy_temp.p_cauchy
#         p_median_temp = cauchy_temp.p_median
#         n_cell = cauchy_temp.n_cell
        
#         p_cauchy_temp_log = -np.log10(p_cauchy_temp)
#         median_log = np.median(p_cauchy_temp_log)
#         IQR_log = np.percentile(p_cauchy_temp_log, 75) - np.percentile(p_cauchy_temp_log, 25)
        
#         w_use = n_cell
#         p_use = p_cauchy_temp
#         if len(p_cauchy_temp) > 15:
#             index = p_cauchy_temp_log < median_log + 2*IQR_log
#             w_use = n_cell[index]
#             p_use = p_cauchy_temp[index]
#             n_remove = len(p_cauchy_temp) - len(p_use)
#             if n_remove > 0:
#                 logger.info(f"Remove {n_remove} outlier (median + 2*IQR) sections for {ct}")
            
#         p_cauchy_new = acat_test(pvalues=p_use.to_list(),weights=w_use.to_list())
#         p_median_new = (p_median_temp * n_cell / n_cell.sum()).sum()
        
#         p_cauchy.append(p_cauchy_new)
#         p_median.append(p_median_new)

#     data = {
#             "p_cauchy": p_cauchy,
#             "p_median": p_median,
#             "annotation": cauchy_all.annotation.unique(),
#             }
#     p_tissue = pd.DataFrame(data)
#     p_tissue.columns = ["p_cauchy", "p_median", "annotation"]
#     p_tissue.sort_values("p_cauchy", inplace=True)
#     return p_tissue


def odds_test_3d(ldsc_merge):
    _, corrected_p_values, _, _ = smm.multipletests(ldsc_merge.p, alpha=0.05)
    ldsc_merge['p_fdr'] = corrected_p_values.tolist()

    Odds = []
    for focal_annotation in ldsc_merge.annotation.unique():
        try:
            focal_no,focal_yes = (ldsc_merge.loc[ldsc_merge.annotation==focal_annotation,'p_fdr'] < 0.05).value_counts()
            other_no,other_yes = (ldsc_merge.loc[ldsc_merge.annotation!=focal_annotation,'p_fdr'] < 0.05).value_counts()
            contingency_table = [[focal_yes, focal_no], [other_yes, other_no]]
            odds_ratio, p_value = fisher_exact(contingency_table)
            table = sm.stats.Table2x2(contingency_table)
            conf_int = table.oddsratio_confint()
        except:
            odds_ratio = 0
            p_value = 1
            conf_int = (0, 0)
        Odds.append({
            'annotation': focal_annotation,
            'odds_ratio': f"{odds_ratio:.3f}",
            '95%_ci_low': f"{conf_int[0]:.3f}",
            '95%_ci_high': f"{conf_int[1]:.3f}",
            'p_odds_ratio': p_value
        })
    Odds = pd.DataFrame(Odds)
    return Odds


def three_d_combine(args: ThreeDCombineConfig):
        
    # Load the ldsc results
    ldsc_merge = combine_ldsc(args)
    ldsc_merge.spot_index = ldsc_merge.spot_index.astype(str).replace(r"\.0", "", regex=True)
    ldsc_merge.index = ldsc_merge.spot_index

    # Load the spatial data
    logger.info(f"Loading {args.adata_3d}.")
    if args.adata_3d.endswith('.parquet'):
        logger.info("The input data is the metadata file of adata.")
        meta_merged = pd.read_parquet(args.adata_3d)
    elif args.adata_3d.endswith('.h5ad'):
        logger.info("The input data is the h5ad.")
        adata_merge = sc.read_h5ad(args.adata_3d,backed='r')
        adata_merge.obs.index.name = 'index'
        spatial = pd.DataFrame(adata_merge.obsm[args.spatial_key], columns=['sx', 'sy', 'sz'], index=adata_merge.obs_names).copy()
        spatial = spatial.reset_index() 
        meta = adata_merge.obs.copy()
        meta_merged = spatial.merge(meta, left_on='index', right_index=True, how='left')
        meta_merged.index = adata_merge.obs_names

    # Handle DataFrame or AnnData
    if args.st_id is not None and (meta_merged.index.value_counts() > 1).any():
        # Check if the index has duplicates and if st_id is provided
        if len(np.intersect1d(ldsc_merge.index, meta_merged.index)) == 0:
            # If no common cells, create a new index using st_id
            logger.info(f"Using {args.st_id} + adata.obs_names as the new cell index.")
            meta_merged.index = meta_merged[args.st_id].astype(str) + '_' + meta_merged.index.astype(str)

    # Find common cells
    common_cell = np.intersect1d(ldsc_merge.index, meta_merged.index)
    if len(common_cell) == 0:
        raise ValueError("No common cells between the spatial data and the ldsc results.")

    logger.info(f"Found {len(common_cell)} common cells between the 3D spatial data and the mapping results.")

    # Subset the data to common cells
    meta_merged = meta_merged.loc[common_cell].copy()
    ldsc_merge = ldsc_merge.loc[common_cell]
    
    # Do cauchy combination test and odds ratio test
    if args.annotation is not None:
        annotation_use = meta_merged[args.annotation]
        ldsc_merge['annotation'] = annotation_use
        ldsc_merge = ldsc_merge[~ldsc_merge.annotation.isna()]
        
        cauchy = cauchy_combination_3d(ldsc_merge)
        odds = odds_test_3d(ldsc_merge)
        cauchy_odds = pd.merge(odds,cauchy,left_on='annotation',right_on='annotation')

        # Save the results
        cauchy_root = Path(args.project_dir) / "3D_combine" / "cauchy_combination"
        cauchy_root.mkdir(parents=True, exist_ok=True, mode=0o755)
        cauchy_name = cauchy_root / f"{args.trait_name}.{args.annotation}.Cauchy.csv.gz"
        cauchy_odds = cauchy_odds.sort_values('odds_ratio',ascending=False)
        cauchy_odds.to_csv(cauchy_name, compression="gzip", index=False)
        logger.info(f"Saving the 3D combination combination results to {cauchy_name}")
    else:
        logger.info("No annotation provided for the cauchy combination test.")

    
    # Plot the 3D results
    p_color = ['#313695', '#4575b4', '#74add1','#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']    
    meta_merged["logp"] = -np.log10(ldsc_merge.p)

    required_columns = {'sx', 'sy', 'sz'}
    if required_columns.issubset(meta_merged.columns):
        logger.info("Generating 3D plot...")
        
        # Set the legend and text
        legend_kwargs = dict(scalar_bar_title_size=30, scalar_bar_label_size=30, fmt="%.1e")
        text_kwargs = dict(text_font_size=15, text_loc="upper_edge")
        
        # Set the opacity for each point
        meta_merged['logp'].fillna(0, inplace=True)
        bins = np.linspace(meta_merged['logp'].min(), meta_merged['logp'].max(), 5)
        alpha = np.exp(np.linspace(0.1, 1.0, num=(len(bins)-1)))-1
        alpha = alpha / max(alpha)
        opacity_show = pd.cut(meta_merged['logp'], bins=bins, labels=alpha, include_lowest=True).values.tolist()   

        # Set the clim
        max_v = np.round(np.median(np.sort(meta_merged['logp'])[::-1][0:20]))
        
        # Plot the 3D results    
        plotter = three_d_plot(
            clim = [0,max_v],
            point_size=args.point_size,
            opacity=opacity_show,
            window_size=(1200, 1008),
            adata=meta_merged,
            spatial_key=args.spatial_key,
            keys=["logp"],
            cmaps=[args.cmap] if args.cmap is not None else [p_color],
            scalar_bar_titles=["-log10(p)"],
            texts=[args.trait_name],
            jupyter=False,
            background=args.background_color,
            show_outline=args.show_outline,
            legend_kwargs=legend_kwargs,
            text_kwargs=text_kwargs,
        )

        # Save the results
        plot_root = Path(args.project_dir) / "3D_combine" / "3D_plot"
        plot_root.mkdir(parents=True, exist_ok=True, mode=0o755)
        plot_name = plot_root / args.trait_name

        three_d_plot_save(
            plotter,
            save_mp4=args.save_mp4,
            save_gif=args.save_gif,
            n_points=args.n_snapshot if args.n_snapshot is not None else 200,
            filename=plot_name,
        )
    else:
        logger.info("The spatial data does not contain 3D spatial coordinates for 3D plotting.")
