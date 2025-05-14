install.packages("devtools")
devtools::install_github("IRkernel/IRkernel")
IRkernel::installspec()

install.packages(c("recipes", "parsnip", "tune", "dials", "workflows", "yardstick", "bonsai", "tidymodels", "tidyverse", "hydroGOF", "lightgbm", "ggthemes", "glue", "tidync", "terra", "dataRetrieval", "tsibble", "zoo", "slider", "sbtools"))
remotes::install_github('mikejohnson51/nwmTools')