# Datasets that need a manual download

These datasets were selected but their hosts cannot be fetched by `waila-download` (no public file API, unsupported
host, or the host timed out). They are in `plan/registry_fixes.json` under `drop`, so `waila-select` backfills other
datasets for their quota. To bring one back, download it by hand into `$WAILA_RAW/<id>/archives/` (or `files/` for loose
images; `waila-download` treats a folder with content as a manual drop and marks it done), remove the id from `drop`,
and re-run `waila-select`.

| id | Host | Where to get it | Notes |
|---|---|---|---|
| `isic-clonal-sk` | ISIC archive | https://api.isic-archive.com/collections/534/ and https://api.isic-archive.com/collections/476/ (DOI https://doi.org/10.34970/231832) | Clonal seborrheic keratoses (PUC Chile) and matched controls; use the ISIC gallery "download" or the `isic-cli` tool. Dermoscopy; license varies per image. |
| `biad2493-two-channel-tirf-colocalization` | BioStudies / BioImage Archive | https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD2493 (DOI https://doi.org/10.6019/S-BIAD2493) | CC0. Two-channel TIRF images; use the study's Globus/FTP/`ascp` download links. |
| `biad3061-gbeta1gamma2-tirf-mobility` | BioStudies / BioImage Archive | https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD3061 (DOI https://doi.org/10.6019/S-BIAD3061) | CC-BY-4.0. Live-cell TIRF. |
| `gzmh-gland` | ScienceDB (scidb.cn) | https://www.scidb.cn/detail?dataSetId=07b63103869f47c1ab05b270ea5a302c (DOI https://doi.org/10.57760/sciencedb.37165) | CC-BY-4.0. Breast gland histology, Ganzhou Municipal Hospital. Browser download (login may be required). |
| `hagen-2021-fluorescence-denoising` | GigaDB | http://gigadb.org/dataset/100888 (DOI https://doi.org/10.5524/100888) | CC0. Hagen et al. 2021 fluorescence denoising; the files are on GigaDB's FTP. |
| `pk-covid-cxr-3568` | Mendeley Data | https://data.mendeley.com/datasets/cmwn37pv98 | CC-BY-4.0. The public API lists no files; use "Download All" in the browser. |
| `ropdb-sfax-mendeley-2024` | Mendeley Data | https://data.mendeley.com/datasets/29w6jd6tvs/2 (DOI https://doi.org/10.17632/29w6jd6tvs.2) | CC-BY-4.0. The public API lists no files; use "Download All" in the browser. |
| `tcga-coad-ct` | TCIA | https://www.cancerimagingarchive.net/collection/tcga-coad/ (DOI https://doi.org/10.7937/K9/TCIA.2016.HJJHBOXZ) | CC-BY-3.0. Needs the NBIA Data Retriever; TCIA is not a supported route (the collection timed out in the pilot). |

No Hugging Face mirror of any of these was found (searched by name on 2026-10-15; the only `tcga-coad` hits are tabular
or pathology-slide repos, not the CT collection).
