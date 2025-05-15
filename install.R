install.packages("devtools")
devtools::install_github("IRkernel/IRkernel")
IRkernel::installspec()

install.packages(c("recipes", "parsnip", "tune", "dials", "workflows", "yardstick", "bonsai", "tidymodels", "tidyverse", "lightgbm", "ggthemes", "glue", "tidync", "terra", "dataRetrieval", "tsibble", "zoo", "slider", "sbtools"))
install.packages(c("dplyr", "httr", "lubridate", "nhdplusTools", "rvest", "terra", "xml2", "profvis"))
remotes::install_github('hzambran/hydroTSM')
remotes::install_github('hzambran/hydroGOF')
remotes::install_github('mikejohnson51/nwmTools')