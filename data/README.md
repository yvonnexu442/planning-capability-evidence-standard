# Public data setup

The source datasets are public but are not redistributed in this repository. Users must obtain them from the original providers and comply with their terms.

## Favorita

Source: Kaggle, *Corporacion Favorita Grocery Sales Forecasting*.

Download the competition files through Kaggle and place the extracted files under `data/raw/favorita/`. The loader expects the sales and supporting calendar/item/store files referenced by `src/data_loaders/favorita_loader.py`. Run the relevant preprocessing or experiment command from the repository root.

## M5

Source: Kaggle, *M5 Forecasting - Accuracy*.

Download and extract the competition files under `data/raw/m5/`. Preserve the original filenames, including the calendar, sales history, and price files expected by `src/data_loaders/m5_loader.py`.

## Walmart

Source: Kaggle, *Walmart Recruiting - Store Sales Forecasting*.

Download and extract the competition files under `data/raw/walmart/`. Preserve the original train, features, and stores filenames expected by `src/data_loaders/walmart_loader.py`.

## Preprocessing and selection

The experiment scripts call the project loaders and apply the frozen chronological splits and panel-selection rules. Do not replace the source files with post-outcome filtered panels. The canonical data identities and source notes are recorded in `public_data_manifest.csv`.

Dataset licenses, competition rules, and redistribution restrictions remain those of the original providers.
