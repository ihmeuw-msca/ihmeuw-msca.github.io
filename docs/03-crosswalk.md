# Crosswalk {#xwalk}



[Example 1, basic crosswalk](#ex1)

[Example 2, network meta-analysis](#ex2)

[Example 3, network meta-analysis with composite groups](#ex3)

[Funnel plots and dose-response plots](#ex4)

---

Crosswalking is the process of taking systematically biased data points and estimating their unbiased value. For example, when estimating overweight prevalence (BMI >= 25), some height and weight observations might be measured physically while others are self-reported. Because people tend to overreport their height and underreport their weight, self-reported BMI is systematically biased downward. Crosswalking involves: 

1. Finding pairs of alternative and reference (e.g. self-reported and measured) observations that match on relevant criteria (e.g. age, sex and location),
2. Taking the difference between these observations in log or logit space, to ensure that the crosswalk adjustment remains bounded correctly,
3. Running a meta-regression model that estimates how this difference varies by covariate values (e.g. age, sex and location); and 
4. Predicting how much the alternative data points in the original dataset should be adjusted. 

To appropriately downweight the adjusted data point, we add uncertainty that comes from crosswalking process itself -- standard error of the prediction and between-group heterogeneity -- to the standard error of the result. 

Sometimes a crosswalk needs to adjust multiple alternative definitions to the gold standard reference. For example, prevalence of schistosomiasis might be measured according to a gold standard diagnostic method A, or three alternatives B, C or D. The process is essentially the same as described above, with the added benefit that _network meta-analysis_ can use information from the indirect comparisons B:C, C:D and B:D. This vignette demonstrates how to use the `crosswalk` package to run a meta-regression (optionally network meta-regression) to predict adjustment factors for biased observations. 

## Example 1: one reference and one alternative {#ex1}

First, we create some simulated data for a crosswalk with one reference and one alternative. In `df_matched`, the variable `logit_diff` represents `logit(prev_alt) - logit(prev_ref)`, or the difference in logit-transformed prevalence values. By convention, we always subtract the reference from the alternative value, alt minus ref. `logit_diff_se` is the standard error of `logit_diff`. The package includes the functions `crosswalk::delta_transform()` and `crosswalk::calculate_diff()` to facilitate calculating these quantities from your matched data. See https://rpubs.com/rsoren/572599 for a full example. 

We also simulate the original dataset containing observations to be adjusted (`df_orig`). The covariates used in the meta-regression model must exist in the original data, with the same column names (`x1` and `x2` in this case). We also include a column that indicates the definition/method used to obtain the observation: `obs_method` with values "selfreported" and "measured" in this case. For `df_orig`, we leave the dependent variable in linear space; the conversion to log or logit space is handled automatically by the `adjust_orig_vals()` function later. 




```r
library(crosswalk002, lib.loc = path_to_r_version4_packages)
library(dplyr)
set.seed(123)
# data for the meta-regression
# -- in a real analysis, you'd get this dataset by 
#    creating matched pairs of alternative/reference observations,
#    then using delta_transform() and calculate_diff() to get
#    log(alt)-log(ref) or logit(alt)-logit(ref) as your dependent variable
beta0_true <- -3
beta1_true <- 1
beta2_true <- 2
df_matched <- data.frame(
  x1 = runif(n = 200, min = 0, max = 4),
  x2 = rbinom(n = 200, prob = 0.5, size = 1) ) %>%
  mutate(
    logit_diff = beta0_true + x1*beta1_true + x2*beta2_true + rnorm(n = nrow(.)),
    logit_diff_se = runif(200, min = 0.9, max = 1.1),
    altvar = "selfreported",
    refvar = "measured",
    group_id = rep(1:20, each = 10)
  )
head(df_matched)
#>         x1 x2 logit_diff logit_diff_se       altvar   refvar group_id
#> 1 1.150310  0  0.3491204     0.9941364 selfreported measured        1
#> 2 3.153221  1  3.4656335     0.9731691 selfreported measured        1
#> 3 1.635908  1  0.3707626     0.9242544 selfreported measured        1
#> 4 3.532070  1  3.0752637     0.9093987 selfreported measured        1
#> 5 3.761869  0  0.3475292     0.9525593 selfreported measured        1
#> 6 0.182226  1 -1.2940209     1.0937282 selfreported measured        1
# original dataset with alternative observations to be adjusted
df_orig <- data.frame(stringsAsFactors = FALSE,
  meanvar = runif(400, min = 0.2, max = 0.8), # original prevalence values; between 0 and 1
  sdvar = runif(400, min = 0.1, max = 0.5), # standard errors of the original prevalence values; >0
  x1 = runif(400, min = 0, max = 4),
  x2 = rbinom(400, prob = 0.5, size = 1),
  obs_method = sample(c("selfreported", "measured"), size = 400, replace = TRUE)
)
df_orig$row_id <- paste0("row", 1:nrow(df_orig))
head(df_orig)
#>     meanvar     sdvar        x1 x2   obs_method row_id
#> 1 0.3641736 0.1932959 0.6210565  1     measured   row1
#> 2 0.5563202 0.1922614 3.3834040  1     measured   row2
#> 3 0.2961109 0.1246905 0.8575217  1 selfreported   row3
#> 4 0.7120581 0.2988474 2.6794930  0     measured   row4
#> 5 0.7086435 0.1976500 2.4710258  0 selfreported   row5
#> 6 0.4867321 0.4032742 0.1999991  1 selfreported   row6
```


Next we use the `crosswalk` functions. `CWData()` prepares the data for meta-regression, and `CWModel()` fits the model. Inputs to the function parameters are described in the comments below, and more information is available in the R package help documentation. Some additional tips...

* When crosswalking one alternative to one reference definition/method, there will be only one value in each of the columns passed to `alt_dorms` and `ref_dorms` (e.g. "selfreported" and "measured", respectively). This may seem like an unnecessary detail now, but it allows for simpler specification of a network meta-analysis within the same framework. More on that in Example 2. 
* The `cov_models` parameter takes a list of `CovModel()` function calls. These specify the functional form and/or constraints for each predictor in the model; see `help(CovModel)` for more information. 
* If you want a model with an intercept, you have to explicitly pass in `CovModel(cov_name = "intercept")` as an element of `cov_models = list(...)`. Note that for network meta-analysis, we almost always *do* want to include an "intercept", which seems at odds with the guidance from GBD 2019. What changed is the definition of intercept, not the model. See the crosswalk training slides (https://rpubs.com/rsoren/572599) for a mathematical explanation. Briefly, "intercepts" refer to what in GBD 2019 were the ${-1,0,1}$ variables that encode the network structure. You no longer have to create these variables yourself, and instead only need to specify `CovModel("intercept")` in a network meta-analysis like the one in Example 2. 
* A continuous covariate can be represented as a spline with the `XSpline()` function. It takes arguments about polynomial degree, linearity in the tails and knot location (must include external knots at <=min and >= max values of the covariate). Relatedly, constraints on spline direction and shape may be specified in the `CovModel()` function with the `spline_monotonicity` and `spline_convexity` arguments. An example is commented out below. 




```r
df1 <- CWData(
  df = df_matched,          # dataset for metaregression
  obs = "logit_diff",       # column name for the observation mean
  obs_se = "logit_diff_se", # column name for the observation standard error
  alt_dorms = "altvar",     # column name of the variable indicating the alternative method
  ref_dorms = "refvar",     # column name of the variable indicating the reference method
  covs = list("x1", "x2"),     # names of columns to be used as covariates later
  study_id = "group_id",    # name of the column indicating group membership, usually the matching groups
  add_intercept = TRUE      # adds a column called "intercept" that may be used in CWModel()
)
fit1 <- CWModel(
  cwdata = df1,            # object returned by `CWData()`
  obs_type = "diff_logit", # "diff_log" or "diff_logit" depending on whether bounds are [0, Inf) or [0, 1]
  cov_models = list(       # specifying predictors in the model; see help(CovModel)
    CovModel(cov_name = "intercept"),
    CovModel(cov_name = "x1"),
    # CovModel(cov_name = "x1", spline = XSpline(knots = c(0,1,2,3,4), degree = 3L, l_linear = TRUE, r_linear = TRUE), spline_monotonicity = "increasing"),
    CovModel(cov_name = "x2") ),
  gold_dorm = "measured"   # the level of `alt_dorms` that indicates it's the gold standard
                           # this will be useful when we can have multiple "reference" groups in NMA
)
print(data.frame( # checking that the model recovers the true betas
  beta_true = c(beta0_true, beta1_true, beta2_true), 
  beta_mean = fit1$beta[4:6],
  beta_se = fit1$beta_sd[4:6]
))
#>   beta_true beta_mean   beta_se
#> 1        -3 -3.164455 0.1628918
#> 2         1  1.081816 0.0640169
#> 3         2  2.076966 0.1412373
```
To create a data frame with estimated coefficients, use `create_result_df()`. This is useful for saving coefficient summaries to CSV. To save a model object for future use, use `py_save_object()`. 

```r
df_result <- fit1$create_result_df()
write.csv(df_result, file.path(path_to_misc_outputs, "df_result_crosswalk.csv"))
py_save_object(object = fit1, filename = file.path(path_to_misc_outputs, "fit1.pkl"), pickle = "dill")
fit1 <- py_load_object(filename = file.path(path_to_misc_outputs, "fit1.pkl"), pickle = "dill")
```




Finally, `adjust_orig_vals()` adjusts biased observations in the original dataset using the meta-regression model to predict the degree of bias. The `adjust_orig_vals()` function has a few requirements:

* While we manually calculated the log- or logit-scale differences for the meta-regression model, for `adjust_orig_vals()` we leave the parameter of interest in linear space.
* For a log-scale crosswalk, `adjust_orig_vals()` cannot make adjustments for observations where the parameter value is zero. This happens because `log(0) = -Inf`. The same happens in logit-scale crosswalks with parameter values of 0 or 1, because `logit(0) = -Inf` and `logit(1) = Inf`. For this reason, `adjust_orig_vals()` will throw an error if the data frame includes observations exactly at the bound(s). 
* Variables used as predictors in the meta-regression model must also be present in dataset passed to `adjust_orig_vals()`, with the same column names. 
* The meta-regression model estimates a random effect for each group specified by the `study_id` variable. Whether you want to predict out on these random effects is a modeling decision that depends on whether you believe the random component captures true variation in the crosswalk adjustment, as opposed to noise in the data. As a default we leave the `adjust_orig_vals(..., study_id = NULL)` parameter unspecified and do not predict out on the random effects. 



```r
preds1 <- adjust_orig_vals(
  fit_object = fit1, # object returned by `CWModel()`
  df = df_orig,
  orig_dorms = "obs_method",
  orig_vals_mean = "meanvar",
  orig_vals_se = "sdvar",
  data_id = "row_id"   # optional argument to add a user-defined ID to the predictions;
                       # name of the column with the IDs
)
# the result of adjust_orig_vals() is a five-element list,
# vectors containing: 
# -- the adjusted mean and SE of the adjusted mean in linear space
# -- the adjustment factor and SE of the adjustment factor in transformed space;
#    note that the adjustment factor is the alt-ref prediction,
#    so we *subtract* this value to make the adjustment
# -- an identifier for the row of the prediction frame the corresponds to the prediction
lapply(preds1, head)
#> $ref_vals_mean
#> [1] 0.3641736 0.5563202 0.3304659 0.7120581 0.7990013 0.6938207
#> 
#> $ref_vals_sd
#> [1] 0.1932959 0.1922614 0.1412219 0.2988474 0.1580049 0.3459730
#> 
#> $pred_diff_mean
#> [1]  0.0000000  0.0000000 -0.1598083  0.0000000 -0.4912590 -0.8711269
#> 
#> $pred_diff_sd
#> [1] 0.0000000 0.0000000 0.2224753 0.0000000 0.2270617 0.2159760
#> 
#> $data_id
#> [1] "row1" "row2" "row3" "row4" "row5" "row6"
# now we add the adjusted values back to the original dataset
df_orig[, 
  c("meanvar_adjusted", "sdvar_adjusted", 
    "pred_logit", "pred_se_logit", "data_id")] <- preds1
# note that the gold standard observations remain untouched
head(df_orig)
#>     meanvar     sdvar        x1 x2   obs_method row_id meanvar_adjusted
#> 1 0.3641736 0.1932959 0.6210565  1     measured   row1        0.3641736
#> 2 0.5563202 0.1922614 3.3834040  1     measured   row2        0.5563202
#> 3 0.2961109 0.1246905 0.8575217  1 selfreported   row3        0.3304659
#> 4 0.7120581 0.2988474 2.6794930  0     measured   row4        0.7120581
#> 5 0.7086435 0.1976500 2.4710258  0 selfreported   row5        0.7990013
#> 6 0.4867321 0.4032742 0.1999991  1 selfreported   row6        0.6938207
#>   sdvar_adjusted pred_logit pred_se_logit data_id
#> 1      0.1932959  0.0000000     0.0000000    row1
#> 2      0.1922614  0.0000000     0.0000000    row2
#> 3      0.1412219 -0.1598083     0.2224753    row3
#> 4      0.2988474  0.0000000     0.0000000    row4
#> 5      0.1580049 -0.4912590     0.2270617    row5
#> 6      0.3459730 -0.8711269     0.2159760    row6
```

To make sure we understand what's going on, let's manually calculate the adjustment for a particular row of the original dataset.


```r
print(df_orig[3,]) # row 3 is a self-reported observation with prevalence 0.296
#>     meanvar     sdvar        x1 x2   obs_method row_id meanvar_adjusted
#> 3 0.2961109 0.1246905 0.8575217  1 selfreported   row3        0.3304659
#>   sdvar_adjusted pred_logit pred_se_logit data_id
#> 3      0.1412219 -0.1598083     0.2224753    row3
fit1$fixed_vars # estimated betas
#> $measured
#> [1] 0 0 0
#> 
#> $selfreported
#> [1] -3.164455  1.081816  2.076966
# the predicted adjustment for an observations with x1=0.8575217 and x2=1 should be...
(pred <- -3.164455 + 1.081816*0.8575217 + 2.076966*1)
#> [1] -0.1598083
# the prediction is defined as logit(alt) - logit(ref), so the final adjusted value should be
# logit(mean_adjusted) = logit(mean_alt) - prediction
logit <- function(p) log(p/(1-p))
inv_logit <- function(x) exp(x)/(1+exp(x))
logit_mean_adjusted <- logit(df_orig[3, "meanvar"]) - pred
inv_logit(logit_mean_adjusted)
#> [1] 0.3304659
# check that it's the same
round(inv_logit(logit_mean_adjusted), digits = 5) == round(df_orig[3, "meanvar_adjusted"], digits = 5)
#> [1] TRUE
# SE of the adjusted data point is calculated as sqrt(a^2 + b^2 + c^2), where
# a is the (log or logit) standard error of the original data point,
# b is the standard error of the predicted adjustment
# c is the standard deviation of between-group heterogeneity, a.k.a. sqrt(gamma)
#
# note that a, b, and c are all in transformed (log or logit) space
# 
# this method increases an adjusted observation's uncertainty and effectively downweights it in subsequent analyses, like an ST-GPR model to estimate prevalence globally
#
```


## Example 2: one reference and multiple alternatives, with indirect comparisons {#ex2}

Network meta-analysis is a method that allows the model to incorporate information from indirect comparisons. For example, in the case below, we have a gold standard diagnostic method A and two alternatives B and C. To adjust B and C observations to their predicted A-equivalents, the best input data we could give the meta-regression is many direct comparisons of B:A and C:A. However, B:C comparisons also help to inform the beta estimates for B:A and C:A, even if B:C is not the quantity of interest. In the example, we specify that the adjustment factor should vary by the level of a continuous covariate `x1`. By excluding intercepts, we ensure that the adjustment factor is 0 when the value of `x1` is 0. 

Note that most network meta-analyses will want to include an intercept term in the list passed to `cov_models`. The model will return a coefficient for each definition that answers the question: here's how much you need to adjust a reference observation to make it equivalent to the alternative definition. As described above, this is different than the guidance in GBD 2019 because the word "intercept" now refers to the ${-1,0,1}$ variables that encode the network. 

First, we create some simulated data. The column indicating which diagnostic is the reference (`refvar`) can include A, B and C. The column indicating the alternative diagnostic (`altvar`) includes only the alternatives, B and C. This means that both B and C can act as a "reference" for an indirect comparison, i.e. the order B:C or C:B doesn't matter.


```r
set.seed(123)
true_beta1 <- 1 # define the true coefficient value for x1
case_defs <- c(B = 2, C = 3) # define case definitions and true coefficient values
df_matched2 <- data.frame(id = 1:400) %>%
  mutate(
    altvar = sample(c("B", "C"), nrow(.), TRUE),
    refvar = sample(c("A", "B", "C"), nrow(.), TRUE),
    x1 = runif(n = nrow(.), min = 0, max = 10),
    logit_diff = rnorm(n = nrow(.)),
    logit_diff = logit_diff + (altvar == "B")*case_defs["B"] * x1*true_beta1,
    logit_diff = logit_diff + (altvar == "C")*case_defs["C"] * x1*true_beta1,
    logit_diff = logit_diff - (refvar == "B")*case_defs["B"] * x1*true_beta1,
    logit_diff = logit_diff - (refvar == "C")*case_defs["C"] * x1*true_beta1,
    # logit_diff_se = runif(n = nrow(.), min = 0.4, max = 0.6),
    logit_diff_se = runif(n = nrow(.), min = 0.4, max = 0.6),
    group_id = sample(1:20, size = nrow(.), replace = TRUE)) %>%
  arrange(group_id) %>%
  filter(altvar != refvar) %>%
  select(altvar, refvar, logit_diff, logit_diff_se, x1, group_id)
head(df_matched2)
#>   altvar refvar logit_diff logit_diff_se       x1 group_id
#> 1      C      B   4.824921     0.5524550 5.394107        1
#> 2      B      C  -3.309311     0.4719643 3.923313        1
#> 3      C      B   4.035348     0.5238145 2.886901        1
#> 4      B      A  18.119626     0.5445521 8.666537        1
#> 5      C      B  10.743544     0.5142872 9.724320        1
#> 6      C      A  24.837169     0.5588720 7.884719        1
```


Next we specify the model the same way as in Example 1, with an important exception. We can give `CWModel()` a prior indicating whether one alternative should give a smaller adjustment than another, using the `order_prior` parameter. It takes a list of two-element vectors, with the first element indicating which diagnostic should have the lower crosswalk adjustment. `order_prior` is useful when one alternative definition is a subset of another. For example, if C is prevalence of current smokers and B is prevalence of daily smokers, B is a subset of C by definition. The default is not to include order priors. 


```r
df2 <- CWData(
  df = df_matched2,         # dataset for metaregression
  obs = "logit_diff",       # column name for the observation mean
  obs_se = "logit_diff_se", # column name for the observation standard error
  alt_dorms = "altvar",     # column name of the variable indicating the alternative method
  ref_dorms = "refvar",     # column name of the variable indicating the reference method
  covs = list("x1"),        # names of columns to be used as covariates later
  # covs = list(),        # names of columns to be used as covariates later
  study_id = "group_id"     # name of the column indicating group membership, usually the matching groups
)
fit2 <- CWModel(
  cwdata = df2,            # object returned by `CWData()`
  obs_type = "diff_logit", # "diff_log" or "diff_logit" depending on whether bounds are [0, Inf) or [0, 1]
  cov_models = list(       # specying predictors in the model; see help(CovModel)
    # CovModel(cov_name = "intercept") ),
    CovModel(cov_name = "x1") ),
  gold_dorm = "A",   # the level of `ref_dorms` that indicates it's the gold standard
  order_prior = list(c("B", "C")) # tells the model that the coefficient estimated for B should be <= the coefficient for C
)
df_tmp <- fit2$create_result_df()
print(data.frame( # checking that the model recovers the true betas
  beta_true = case_defs,
  beta_mean = fit2$beta[2:3],
  beta_se = fit2$beta_sd[2:3]
))
#>   beta_true beta_mean     beta_se
#> B         2  2.000413 0.009301158
#> C         3  3.002995 0.009994177
```

Finally, as before, we use the model to adjust biased observations in the original dataset. 


```r
df_orig2 <- data.frame(id = 1:600) %>%
  mutate(
    meanvar = runif(n = nrow(.), min = 0.2, max = 0.8),
    sdvar = runif(n = nrow(.), min = 0.4, max = 0.6),
    x1 = runif(n = nrow(.), min = 0, max = 10),
    group_id = sample(1:20, size = nrow(.), replace = TRUE),
    intercept = 0,
    obs_method = sample(c("A", "B", "C"), nrow(.), TRUE) )
head(df_orig2)
#>   id   meanvar     sdvar       x1 group_id intercept obs_method
#> 1  1 0.5135624 0.5610527 8.169894       11         0          A
#> 2  2 0.6794278 0.5537111 7.596990       11         0          C
#> 3  3 0.3244733 0.4681815 5.718239       20         0          C
#> 4  4 0.4317691 0.5875787 9.887544        7         0          A
#> 5  5 0.5254297 0.5007663 9.951715       14         0          C
#> 6  6 0.6465518 0.4648555 3.214374       19         0          C
preds2 <- adjust_orig_vals(
  fit_object = fit2, # object returned by `CWModel()`
  df = df_orig2,
  orig_dorms = "obs_method",
  orig_vals_mean = "meanvar",
  orig_vals_se = "sdvar"
)
# now we add the adjusted values back to the original dataset
df_orig2[, c(
  "meanvar_adjusted", "sdvar_adjusted", "pred_logit", 
  "pred_se_logit", "data_id")] <- preds2
# note that the gold standard observations remain untouched
head(df_orig2)
#>   id   meanvar     sdvar       x1 group_id intercept obs_method
#> 1  1 0.5135624 0.5610527 8.169894       11         0          A
#> 2  2 0.6794278 0.5537111 7.596990       11         0          C
#> 3  3 0.3244733 0.4681815 5.718239       20         0          C
#> 4  4 0.4317691 0.5875787 9.887544        7         0          A
#> 5  5 0.5254297 0.5007663 9.951715       14         0          C
#> 6  6 0.6465518 0.4648555 3.214374       19         0          C
#>   meanvar_adjusted sdvar_adjusted pred_logit pred_se_logit data_id
#> 1     5.135624e-01   5.611872e-01   0.000000    0.00000000       0
#> 2     2.620254e-10   6.665493e-10  22.813723    0.07592566       1
#> 3     1.674553e-08   3.579003e-08  17.171845    0.05714910       2
#> 4     4.317691e-01   5.877026e-01   0.000000    0.00000000       3
#> 5     1.162370e-13   2.337901e-13  29.884952    0.09945921       4
#> 6     1.175142e-04   2.391163e-04   9.652751    0.03212503       5
```


## Example 3: one reference and multiple alternatives; alternatives composed of sub-definitions {#ex3}

Sometimes alternative definitions are not mutually exclusive. For example, maybe the gold standard way to measure prevalence is with diagnostic tool A and among the general population. However, some observations measured prevalence with diagnostic tool B and/or only in urban populations (C). In addition to using the non-standard diagnostic tool in the general population (denoted as "B"), alternative observations can take on values of "C" or "B_C" if the prevalence was measured in an urban population. We account for this in a crosswalk adjustment by assuming additivity of the sub-definitions, i.e. the logit-space adjustment due to taking prevalence in an urban population is the same regardless of which diagnostic tool was used. As in Example 2, we relax the requirement that prevalence comparisons must include the gold-standard reference. Network meta-analysis allows the model to take advantage of comparisons between two alternative definitions, e.g. "B" and "B_C".


```r
set.seed(1)
df3 <- data.frame(id = 1:400)
#####
# create the simulated dataset
# define case definitions and coefficients for simulation
case_defs <- c(B = 2, C = 3)
n_sample_defs <- 2
# randomly sample from case definition components to make combined case definitions
df3$alt <- sapply(1:nrow(df3), function(i) {
  paste(sort(sample(x = names(case_defs), size = sample(1:n_sample_defs))), collapse = '_')
})
df3$ref <- sapply(1:nrow(df3), function(i) {
  paste(sort(sample(x = names(case_defs), size = sample(1:n_sample_defs))), collapse = '_')
})
# make half of the reference case definitions 'A' to signify the gold standard
df3$ref <- ifelse(df3$id %% 2 == 0, "A", df3$ref)
df4 <- filter(df3, ref != alt) # remove duplicates
# subtract coefficient if component is in the reference def, and 
# add coefficient if component is in the alternative def
df4$logit_prev_diff <- 0
for (i in names(case_defs)) df4$logit_prev_diff <- df4$logit_prev_diff - (sapply(i, grepl, df4$ref) * case_defs[i])
for (i in names(case_defs)) df4$logit_prev_diff <- df4$logit_prev_diff + (sapply(i, grepl, df4$alt) * case_defs[i])
df4$logit_prev_diff <- as.numeric(df4$logit_prev_diff + rnorm(n = nrow(df4), mean = 0, sd = 1))
df4$logit_prev_diff_se <- 0.5
head(df4)
#>   id alt ref logit_prev_diff logit_prev_diff_se
#> 1  1   B B_C       -1.095623                0.5
#> 2  2   B   A        1.651368                0.5
#> 3  3   C   B        1.218831                0.5
#> 4  4 B_C   A        6.144955                0.5
#> 5  5   C B_C       -2.123546                0.5
#> 6  6 B_C   A        5.147393                0.5
```

To use composite definitions -- alternative definitions with non-mutually exclusive sub-definitions -- we include a delimiter in the values we pass to the `alt_dorms` and `ref_dorms` columns. If the delimiter is "_" as in this example, we pass it as an argument to the `dorm_separator` parameter of `CWData()`. 


```r
#####
dat4 <- CWData(
  df = df4,
  obs = "logit_prev_diff",
  obs_se = "logit_prev_diff_se",
  alt_dorms = "alt",
  ref_dorms = "ref",
  dorm_separator = "_",
  covs = list(),
  study_id = "id",
  add_intercept = TRUE
)
fit4 <- CWModel(
  cwdata = dat4,
  obs_type = "diff_logit",
  cov_models = list(CovModel("intercept")),
  # # we can put priors on the beta for a specific alternative definition:
  # cov_models = list(CovModel("intercept", prior_beta_uniform = list(B = array(c(0,0))))),
  gold_dorm = "A"
  # prior_gamma_uniform = array(c(0, 0.4))
)
fit4$fixed_vars
#> $A
#> [1] 0
#> 
#> $B
#> [1] 2.002939
#> 
#> $C
#> [1] 3.033783
case_defs
#> B C 
#> 2 3
```

```r
df_orig4 <- df4 %>%
  mutate(
    prev_orig = runif(n = nrow(.), 0.1, 0.9),
    prev_orig_se = 0.2,
    meas_method = ref) %>%
  select(-ref, -alt, -logit_prev_diff, -logit_prev_diff_se)
preds4 <- adjust_orig_vals(
  fit_object = fit4, # object returned by `CWModel()`
  df = df_orig4,
  orig_dorms = "meas_method",
  orig_vals_mean = "prev_orig",
  orig_vals_se = "prev_orig_se"
)
df_orig4[, c("prev_adjusted", "prev_se_adjusted", "prediction_logit", 
             "prediction_se_logit", "data_id")] <- preds4
head(df_orig4)
#>   id prev_orig prev_orig_se meas_method prev_adjusted prev_se_adjusted
#> 1  1 0.6294998          0.2         B_C    0.01091491       0.01325531
#> 2  2 0.1380010          0.2           A    0.13800096       0.22532907
#> 3  3 0.8047976          0.2           B    0.35746479       0.35488945
#> 4  4 0.8455840          0.2           A    0.84558400       0.23017319
#> 5  5 0.8639310          0.2         B_C    0.03960492       0.07283514
#> 6  6 0.7042963          0.2           A    0.70429629       0.27022408
#>   prediction_logit prediction_se_logit data_id
#> 1         5.036722          0.10433740       0
#> 2         0.000000          0.00000000       1
#> 3         2.002939          0.07302781       2
#> 4         0.000000          0.00000000       3
#> 5         5.036722          0.10433740       4
#> 6         0.000000          0.00000000       5
```


## Funnel plots and dose-response plots {#ex4}

To assess the fit of a crosswalk model, we use a funnel plot to see how well a single crosswalk coefficient describes the data. When the crosswalk adjustment varies by levels of a continuous variable, we use a dose-response plot. First, to enable the use of the `matplotlib` library in the underlying Python code, we run `repl_python()` to open an interactive Python interpreter, then type `exit` in the console to return to the R interpreter. (We're still figuring out why this is necessary and will fix it when we do!)

If any covariates were included in the meta-regression model, the `plots$funnel_plot()` and `plots$dose_response_curve()` functions assume a covariate value of zero unless otherwise specified by the `continuous_variables` parameter. The prediction will be based upon the median value observed in the input data for all variables names passed to `continuous_variables`. `obs_method` indicates which alternative definition/method (dorm) should be visualized, compared to the gold standard reference. This parameter should be specified regardless of whether or not there are multiple alternative definitions. 

PDFs of the plots are saved in the location passed to `plots_dir`.



```r
df_matched5 <- data.frame(id = 1:200) %>%
  mutate(
    logit_diff_se = runif(nrow(.), 0.5, 10),
    logit_diff = 5 + rnorm(nrow(.), 0, logit_diff_se),
    altvar = ifelse(row_number() %% 8 == 0, "awefawef", "selfreported"),
    refvar = "measured",
    group_id = rep(1:10, each = 20)
  )
dat5 <- CWData(
  df = df_matched5,
  obs = "logit_diff",
  obs_se = "logit_diff_se",
  alt_dorms = "altvar",
  ref_dorms = "refvar",
  covs = list(),
  study_id = "group_id",
  add_intercept = TRUE
)
fit5 <- CWModel(
  cwdata = dat5,
  obs_type = "diff_logit",
  cov_models = list(
    CovModel(cov_name = "intercept") ),
  gold_dorm = "measured",
  inlier_pct = 0.9
)
##### don't forget to run repl_python() !
# ... then type 'exit' to get back to the R interpreter
repl_python()
plots <- import("crosswalk.plots")
plots$funnel_plot(
  cwmodel = fit5, 
  cwdata = dat5,
  continuous_variables = list(),
  obs_method = 'selfreported',
  plot_note = 'Funnel plot example', 
  plots_dir = path_to_misc_outputs, 
  file_name = "funny_plot_example_v10",
  write_file = TRUE
)
```

For the dose-response plot, the variable given to `dose_variable` will be on the x-axis.


```r
plots$dose_response_curve(
  dose_variable = 'x1',
  obs_method = 'B', 
  continuous_variables=list(), 
  cwdata=df2, 
  cwmodel=fit2, 
  plot_note="Example dose-response plot", 
  plots_dir=path_to_misc_outputs, 
  file_name = "doseresponse_plot_example_v7", 
  write_file=TRUE)
# py_run_string("import importlib; importlib.reload(plots)")
# importlib <- import("importlib")
# plots <- importlib$reload(plots)
```





